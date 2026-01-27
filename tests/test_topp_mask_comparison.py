"""Test to find differences between int dtype topp_mask and vLLM f32 topp_mask."""

import jax
import jax.numpy as jnp
from tallax.vllm.topp_mask import topp_mask as int_topp_mask
from tallax.vllm.tpu_inference_sampling_as_standalone_file import topp_mask as f32_topp_mask


def test_topp_masks_match():
    """Find cases where int dtype and f32 topp_mask give different results."""

    # Test with various configurations to find a mismatch
    configs = [
        (8, 1024, 42, 0.9),
        (8, 1024, 123, 0.5),
        (8, 1024, 456, 0.95),
        (16, 2048, 42, 0.9),
        (4, 512, 42, 0.8),
    ]

    for batch_size, vocab_size, seed, p_val in configs:
        key = jax.random.PRNGKey(seed)
        logits_key, p_key = jax.random.split(key, 2)

        # Generate random logits
        logits = jax.random.normal(logits_key, (batch_size, vocab_size)).astype(jnp.float32)

        # Test with scalar p
        p_scalar = p_val

        # Apply all implementations
        from tallax.vllm.topp_mask import topp_mask_pallas
        int_result = int_topp_mask(logits, p_scalar, replace_val=-1e12)
        pallas_result = topp_mask_pallas(logits, p_scalar, replace_val=-1e12, interpret=True)
        f32_result = f32_topp_mask(logits, p_scalar, replace_val=-1e12, stable=False)

        # Compare masks
        int_mask = (int_result == -1e12)
        pallas_mask = (pallas_result == -1e12)
        f32_mask = (f32_result == -1e12)
        
        # print(jnp.stack([jnp.where(res == -1e12, jnp.inf, res).min(1) for res in (int_result, pallas_result, f32_result)], axis=1))

        masks_equal = jnp.array_equal(int_mask, f32_mask) and jnp.array_equal(pallas_mask, f32_mask)

        if not masks_equal:
            print(f"\n{'='*70}")
            print(f"FOUND MISMATCH!")
            print(f"Config: batch_size={batch_size}, vocab_size={vocab_size}, seed={seed}, p={p_val}")
            print(f"{'='*70}")

            # Find which batch elements differ
            diff_per_batch = (int_mask != f32_mask).sum(axis=1)
            differing_batches = jnp.where(diff_per_batch > 0)[0]

            print(f"Batches with differences: {differing_batches}")

            # Focus on first differing batch
            if len(differing_batches) > 0:
                b = int(differing_batches[0])
                print(f"\nAnalyzing batch {b}:")
                print(f"  Int masked count: {int_mask[b].sum()}")
                print(f"  F32 masked count: {f32_mask[b].sum()}")
                print(f"  Difference count: {(int_mask[b] != f32_mask[b]).sum()}")

                # Find differing indices
                diff_indices = jnp.where(int_mask[b] != f32_mask[b])[0]
                print(f"  First 10 differing indices: {diff_indices[:10]}")

                # Show the logits at those indices
                for i in diff_indices[:3]:
                    print(f"\n  Index {i}:")
                    print(f"    Logit value: {logits[b, i]}")
                    print(f"    Int masked: {int_mask[b, i]}")
                    print(f"    F32 masked: {f32_mask[b, i]}")

                # Create a minimal test case
                create_minimal_test_case(logits[b:b+1], p_scalar, b, seed)

            return False

    print("\nAll configs matched!")
    return True


def create_minimal_test_case(logits, p, batch_idx, seed):
    """Create a minimal reproducible test case."""
    print(f"\n{'='*70}")
    print("MINIMAL TEST CASE")
    print(f"{'='*70}")

    print(f"""
# Minimal test to reproduce the mismatch
import jax
import jax.numpy as jnp
from tallax.vllm.topp_mask import topp_mask as int_topp_mask
from tallax.vllm.tpu_inference_sampling_as_standalone_file import topp_mask as f32_topp_mask

# Single batch element test
seed = {seed}
batch_idx = {batch_idx}
p = {p}
vocab_size = {logits.shape[1]}

# Generate logits (using same seed to reproduce)
key = jax.random.PRNGKey(seed)
logits_key, _ = jax.random.split(key, 2)
all_logits = jax.random.normal(logits_key, ({logits.shape[0] + batch_idx}, vocab_size)).astype(jnp.float32)
logits = all_logits[{batch_idx}:{batch_idx+1}]  # Shape (1, vocab_size)

# Apply both implementations
int_result = int_topp_mask(logits, p, replace_val=-1e12)
f32_result = f32_topp_mask(logits, p, replace_val=-1e12, stable=False)

# Compare
int_mask = (int_result == -1e12)
f32_mask = (f32_result == -1e12)
print(f"Masks equal: {{jnp.array_equal(int_mask, f32_mask)}}")
print(f"Int masked count: {{int_mask.sum()}}")
print(f"F32 masked count: {{f32_mask.sum()}}")
print(f"Difference count: {{(int_mask != f32_mask).sum()}}")
""")


if __name__ == "__main__":
    print("Searching for differences between int dtype and f32 topp_mask...\n")
    test_topp_masks_match()
