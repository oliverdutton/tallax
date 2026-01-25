"""Test to find differences in the full topk + topp masking pipeline."""

import jax
import jax.numpy as jnp
from tallax.vllm.topk_mask import topk_mask_ref_inputs
from tallax.vllm.topp_mask import topp_mask as int_topp_mask
from tallax.vllm.tpu_inference_sampling_as_standalone_file import (
    topk_mask as f32_topk_mask,
    topp_mask as f32_topp_mask,
)


def test_full_pipeline():
    """Test the full topk + topp masking pipeline."""

    # Use the exact same config as the failing test
    batch_size, vocab_size = 8, 1024
    seed = 42

    key = jax.random.PRNGKey(seed)
    key, topk_key, topp_key, temp_key, logits_key, sample_key = jax.random.split(key, 6)

    # Create varying sampling parameters (same as test)
    k = jax.random.randint(topk_key, (batch_size,), 1, 64, dtype=jnp.int32)
    p = jax.random.uniform(topp_key, (batch_size,), dtype=jnp.float32, minval=0.5, maxval=1.0)
    temperature = jnp.ones((batch_size,), dtype=jnp.float32)

    # Generate random logits
    logits = jax.random.normal(logits_key, (batch_size, vocab_size)).astype(jnp.float32)

    print("Testing full pipeline (topk + topp masking)...")
    print(f"k values: {k}")
    print(f"p values: {p}")

    # Test each batch element separately
    for b in range(batch_size):
        logit_b = logits[b:b+1, :]
        k_b = k[b]
        p_b = p[b]

        print(f"\n{'='*70}")
        print(f"Batch {b}: k={k_b}, p={p_b:.4f}")
        print(f"{'='*70}")

        # Pallas pipeline
        pallas_topk = topk_mask_ref_inputs(
            logit_b,
            jnp.array([[k_b]], dtype=jnp.int32),
            replace_val=-1e12,
            stable=True
        )
        pallas_topp = int_topp_mask(
            pallas_topk,
            p_b,
            replace_val=-1e12
        )

        # Reference pipeline
        ref_topk = f32_topk_mask(
            logit_b.squeeze(),
            k_b,
            replace_val=-1e12,
            stable=True
        ).reshape(1, -1)
        ref_topp = f32_topp_mask(
            ref_topk,
            p_b,
            replace_val=-1e12,
            stable=True
        )

        # Compare after topk
        topk_masks_equal = jnp.array_equal(
            pallas_topk == -1e12,
            ref_topk == -1e12
        )
        print(f"  TopK masks equal: {topk_masks_equal}")

        if not topk_masks_equal:
            pallas_topk_mask = (pallas_topk == -1e12)
            ref_topk_mask = (ref_topk == -1e12)
            print(f"  Pallas topk masked: {pallas_topk_mask.sum()}")
            print(f"  Ref topk masked: {ref_topk_mask.sum()}")
            diff_count = (pallas_topk_mask != ref_topk_mask).sum()
            print(f"  Difference count: {diff_count}")

            # Find differing indices
            diff_indices = jnp.where(pallas_topk_mask[0] != ref_topk_mask[0])[0]
            print(f"  First 10 differing indices: {diff_indices[:10]}")

            # Show logit values at those indices
            for i in diff_indices[:5]:
                print(f"\n    Index {i}:")
                print(f"      Logit: {logit_b[0, i]:.6f}")
                print(f"      Pallas masked: {pallas_topk_mask[0, i]}")
                print(f"      Ref masked: {ref_topk_mask[0, i]}")

        # Compare after topp
        topp_masks_equal = jnp.array_equal(
            pallas_topp == -1e12,
            ref_topp == -1e12
        )
        print(f"  TopP masks equal: {topp_masks_equal}")

        if not topp_masks_equal:
            pallas_topp_mask = (pallas_topp == -1e12)
            ref_topp_mask = (ref_topp == -1e12)
            print(f"  Pallas topp masked: {pallas_topp_mask.sum()}")
            print(f"  Ref topp masked: {ref_topp_mask.sum()}")

        # Sample from both
        pallas_temp = pallas_topp / temperature[b]
        ref_temp = ref_topp / temperature[b]

        pallas_sample = jax.random.categorical(sample_key, pallas_temp)
        ref_sample = jax.random.categorical(sample_key, ref_temp)

        print(f"  Pallas sample: {pallas_sample}")
        print(f"  Ref sample: {ref_sample}")
        print(f"  Samples match: {pallas_sample == ref_sample}")

        if pallas_sample != ref_sample:
            print(f"\n  MISMATCH FOUND!")
            create_isolated_test(
                logits, k, p, temperature, sample_key, b, seed
            )
            break


def create_isolated_test(logits, k, p, temperature, sample_key, batch_idx, seed):
    """Create an isolated test case for the mismatch."""
    print(f"\n{'='*70}")
    print("ISOLATED TEST CASE")
    print(f"{'='*70}")

    print(f"""
# Minimal test to reproduce sampling mismatch
import jax
import jax.numpy as jnp
from tallax.vllm.topk_mask import topk_mask_ref_inputs
from tallax.vllm.topp_mask import topp_mask as int_topp_mask
from tallax.vllm.tpu_inference_sampling_as_standalone_file import (
    topk_mask as f32_topk_mask,
    topp_mask as f32_topp_mask,
)

seed = {seed}
batch_idx = {batch_idx}
vocab_size = {logits.shape[1]}

# Regenerate exact same data
key = jax.random.PRNGKey(seed)
key, topk_key, topp_key, temp_key, logits_key, sample_key = jax.random.split(key, 6)

k_all = jax.random.randint(topk_key, (8,), 1, 64, dtype=jnp.int32)
p_all = jax.random.uniform(topp_key, (8,), dtype=jnp.float32, minval=0.5, maxval=1.0)
logits_all = jax.random.normal(logits_key, (8, vocab_size)).astype(jnp.float32)

# Extract the specific batch element
logits = logits_all[{batch_idx}:{batch_idx+1}]
k_val = k_all[{batch_idx}]
p_val = p_all[{batch_idx}]
temp_val = 1.0

print(f"Testing batch {{batch_idx}}: k={{k_val}}, p={{p_val:.4f}}")

# Pallas pipeline
pallas_topk = topk_mask_ref_inputs(
    logits, jnp.array([[k_val]], dtype=jnp.int32),
    replace_val=-1e12, stable=True
)
pallas_topp = int_topp_mask(pallas_topk, p_val, replace_val=-1e12)
pallas_temp = pallas_topp / temp_val
pallas_sample = jax.random.categorical(sample_key, pallas_temp)

# Reference pipeline
ref_topk = f32_topk_mask(
    logits.squeeze(), k_val, replace_val=-1e12, stable=True
).reshape(1, -1)
ref_topp = f32_topp_mask(ref_topk, p_val, replace_val=-1e12, stable=True)
ref_temp = ref_topp / temp_val
ref_sample = jax.random.categorical(sample_key, ref_temp)

print(f"Pallas sample: {{pallas_sample}}")
print(f"Ref sample: {{ref_sample}}")
print(f"Match: {{pallas_sample == ref_sample}}")

# Debug: compare masks
print(f"\\nTopK masks equal: {{jnp.array_equal(pallas_topk == -1e12, ref_topk == -1e12)}}")
print(f"TopP masks equal: {{jnp.array_equal(pallas_topp == -1e12, ref_topp == -1e12)}}")
""")


if __name__ == "__main__":
    test_full_pipeline()
