"""Isolated test case showing mismatch between Pallas and reference implementations.

This test reproduces a specific case where the Pallas topk_topp_mask_and_sample
implementation gives a different sampled token than the reference vLLM implementation.

The mismatch occurs at:
- Batch index: 1
- k=15, p=0.860743
- Pallas samples: 894
- Reference samples: 566

Both use the same RNG key, so the difference must be in the masking logic.
"""

import jax
import jax.numpy as jnp
from tallax.vllm.topk_topp_mask_and_sample import topk_topp_mask_and_sample
from tallax.vllm.tpu_inference_sampling_as_standalone_file import (
    topk_mask,
    topp_mask,
    _SAMPLING_EPS,
)


def reference_impl(logits, rng_key, k, p, temperature, *, stable=True, replace_val=-1e12):
    """Reference implementation using standalone vLLM functions."""
    batch_size = logits.shape[0]
    k = jnp.broadcast_to(k, (batch_size,))
    p = jnp.broadcast_to(p, (batch_size,))
    temperature = jnp.broadcast_to(temperature, (batch_size,))
    greedy_sampled = jnp.argmax(logits, axis=-1)
    logits = logits.astype(jnp.float32)
    logits_masked = jax.vmap(
        lambda l, k_val: topk_mask(l, k_val, replace_val=replace_val, stable=stable)
    )(logits, k)
    logits_masked = jax.vmap(
        lambda l, p_val: topp_mask(l, p_val, replace_val=replace_val, stable=False)
    )(logits_masked, p)
    temperature_expanded = jnp.expand_dims(temperature, axis=-1)
    logits_masked = logits_masked / temperature_expanded.astype(logits_masked.dtype)
    next_tokens = jax.random.categorical(rng_key, logits_masked)
    return jnp.where(temperature < _SAMPLING_EPS, greedy_sampled, next_tokens)


def test_isolated_mismatch():
    """Test the isolated mismatch case."""
    # Reproduce exact data
    seed = 42
    batch_idx = 1

    key = jax.random.PRNGKey(seed)
    key, topk_key, topp_key, temp_key, logits_key, sample_key = jax.random.split(key, 6)

    k_all = jax.random.randint(topk_key, (8,), 1, 64, dtype=jnp.int32)
    p_all = jax.random.uniform(
        topp_key, (8,), dtype=jnp.float32, minval=0.5, maxval=1.0
    )
    temperature_all = jnp.ones((8,), dtype=jnp.float32)
    logits_all = jax.random.normal(logits_key, (8, 1024)).astype(jnp.float32)

    # Extract single batch element
    logits = logits_all[batch_idx : batch_idx + 1]
    k = k_all[batch_idx : batch_idx + 1]
    p = p_all[batch_idx : batch_idx + 1]
    temperature = temperature_all[batch_idx : batch_idx + 1]

    print(f"Testing isolated batch {batch_idx}: k={k[0]}, p={p[0]:.6f}")

    # Test both implementations
    pallas = topk_topp_mask_and_sample(
        logits, sample_key, k, p, temperature, stable=True, block_token=8, interpret=True
    )
    reference = reference_impl(logits, sample_key, k, p, temperature, stable=True)

    print(f"\nPallas sampled token:    {pallas[0]}")
    print(f"Reference sampled token: {reference[0]}")
    print(f"Match: {pallas[0] == reference[0]}")

    # This test currently fails - the implementations differ
    # TODO: Debug why the masking gives different results
    assert pallas[0] == reference[0], (
        f"Mismatch: Pallas={pallas[0]}, Reference={reference[0]}"
    )


if __name__ == "__main__":
    try:
        test_isolated_mismatch()
        print("\n✓ Test passed!")
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        exit(1)
