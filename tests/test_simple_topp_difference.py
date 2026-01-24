"""Simple test to isolate topp_mask difference causing sampling mismatch."""

import jax
import jax.numpy as jnp
from tallax.vllm.topk_topp_mask_and_sample import topk_topp_mask_and_sample
from tallax.vllm.tpu_inference_sampling_as_standalone_file import (
    topk_mask,
    topp_mask,
    _SAMPLING_EPS,
)


def reference_topk_topp_mask_and_sample(
    logits, rng_key, k, p, temperature, *, stable=True, replace_val=-1e12
):
    """Reference implementation using standalone vLLM functions."""
    batch_size = logits.shape[0]
    k = jnp.broadcast_to(k, (batch_size,))
    p = jnp.broadcast_to(p, (batch_size,))
    temperature = jnp.broadcast_to(temperature, (batch_size,))

    greedy_sampled = jnp.argmax(logits, axis=-1)
    logits = logits.astype(jnp.float32)

    # Apply top-k masking
    logits_masked = jax.vmap(
        lambda l, k_val: topk_mask(l, k_val, replace_val=replace_val, stable=stable)
    )(logits, k)

    # Apply top-p masking (stable=False for topp, only topk uses stable)
    logits_masked = jax.vmap(
        lambda l, p_val: topp_mask(l, p_val, replace_val=replace_val, stable=False)
    )(logits_masked, p)

    # Apply temperature
    temperature_expanded = jnp.expand_dims(temperature, axis=-1)
    logits_masked = logits_masked / temperature_expanded.astype(logits_masked.dtype)

    # Sample
    next_tokens = jax.random.categorical(rng_key, logits_masked)

    return jnp.where(temperature < _SAMPLING_EPS, greedy_sampled, next_tokens)


def test_find_difference():
    """Find the exact batch element where sampling differs."""

    batch_size, vocab_size = 8, 1024
    seed = 42

    key = jax.random.PRNGKey(seed)
    key, topk_key, topp_key, temp_key, logits_key, sample_key = jax.random.split(key, 6)

    k = jax.random.randint(topk_key, (batch_size,), 1, 64, dtype=jnp.int32)
    p = jax.random.uniform(
        topp_key, (batch_size,), dtype=jnp.float32, minval=0.5, maxval=1.0
    )
    temperature = jnp.ones((batch_size,), dtype=jnp.float32)
    logits = jax.random.normal(logits_key, (batch_size, vocab_size)).astype(jnp.float32)

    print(f"k: {k}")
    print(f"p: {p}")

    # Run both implementations
    pallas_result = topk_topp_mask_and_sample(
        logits, sample_key, k, p, temperature, stable=True, block_token=8, interpret=True
    )
    reference_result = reference_topk_topp_mask_and_sample(
        logits, sample_key, k, p, temperature, stable=True
    )

    print(f"\nPallas result:    {pallas_result}")
    print(f"Reference result: {reference_result}")
    print(f"Match: {jnp.array_equal(pallas_result, reference_result)}")

    # Find which batch elements differ
    diff_mask = pallas_result != reference_result
    diff_indices = jnp.where(diff_mask)[0]

    if len(diff_indices) > 0:
        print(f"\nDifferences at batch indices: {diff_indices}")

        # Focus on first difference
        b = int(diff_indices[0])
        print(f"\n{'='*70}")
        print(f"ANALYZING BATCH {b}")
        print(f"{'='*70}")
        print(f"k={k[b]}, p={p[b]:.6f}, temperature={temperature[b]}")
        print(f"Pallas sampled: {pallas_result[b]}")
        print(f"Reference sampled: {reference_result[b]}")

        # Create isolated test
        create_isolated_test(b, seed)

    else:
        print("\nNo differences found!")


def create_isolated_test(batch_idx, seed):
    """Create isolated test case."""
    print(f"""
# Save this as a standalone test file
import jax
import jax.numpy as jnp
from tallax.vllm.topk_topp_mask_and_sample import topk_topp_mask_and_sample
from tallax.vllm.tpu_inference_sampling_as_standalone_file import (
    topk_mask, topp_mask, _SAMPLING_EPS
)

def reference_impl(logits, rng_key, k, p, temperature, *, stable=True, replace_val=-1e12):
    batch_size = logits.shape[0]
    k = jnp.broadcast_to(k, (batch_size,))
    p = jnp.broadcast_to(p, (batch_size,))
    temperature = jnp.broadcast_to(temperature, (batch_size,))
    greedy_sampled = jnp.argmax(logits, axis=-1)
    logits = logits.astype(jnp.float32)
    logits_masked = jax.vmap(lambda l, k_val: topk_mask(l, k_val, replace_val=replace_val, stable=stable))(logits, k)
    logits_masked = jax.vmap(lambda l, p_val: topp_mask(l, p_val, replace_val=replace_val, stable=False))(logits_masked, p)
    temperature_expanded = jnp.expand_dims(temperature, axis=-1)
    logits_masked = logits_masked / temperature_expanded.astype(logits_masked.dtype)
    next_tokens = jax.random.categorical(rng_key, logits_masked)
    return jnp.where(temperature < _SAMPLING_EPS, greedy_sampled, next_tokens)

# Reproduce exact data
seed = {seed}
batch_idx = {batch_idx}

key = jax.random.PRNGKey(seed)
key, topk_key, topp_key, temp_key, logits_key, sample_key = jax.random.split(key, 6)

k_all = jax.random.randint(topk_key, (8,), 1, 64, dtype=jnp.int32)
p_all = jax.random.uniform(topp_key, (8,), dtype=jnp.float32, minval=0.5, maxval=1.0)
temperature_all = jnp.ones((8,), dtype=jnp.float32)
logits_all = jax.random.normal(logits_key, (8, 1024)).astype(jnp.float32)

# Extract single batch element
logits = logits_all[batch_idx:batch_idx+1]
k = k_all[batch_idx:batch_idx+1]
p = p_all[batch_idx:batch_idx+1]
temperature = temperature_all[batch_idx:batch_idx+1]

print(f"Testing isolated batch {{batch_idx}}: k={{k[0]}}, p={{p[0]:.6f}}")

# Test
pallas = topk_topp_mask_and_sample(logits, sample_key, k, p, temperature, stable=True, block_token=8, interpret=True)
reference = reference_impl(logits, sample_key, k, p, temperature, stable=True)

print(f"Pallas: {{pallas[0]}}")
print(f"Reference: {{reference[0]}}")
print(f"Match: {{pallas[0] == reference[0]}}")
""")


if __name__ == "__main__":
    test_find_difference()
