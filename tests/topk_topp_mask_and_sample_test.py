"""Tests for topk_topp_mask_and_sample Pallas kernel.

This test suite validates the combined top-k, top-p masking and sampling kernel against:
1. jax.lax.top_k for top-k masking (f32 and stable)
2. Reference implementation from tpu_inference_sampling for top-p
3. jax.random.categorical for sampling
4. Exact match on sampled tokens
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from tallax.vllm.topk_topp_mask_and_sample import topk_topp_mask_and_sample
from tallax.vllm.tpu_inference_sampling_as_standalone_file import (
  topk_mask,
  topp_mask,
  _SAMPLING_EPS,
)
from tallax.tax.utils import is_cpu_platform


# Helper to call topk_topp_mask_and_sample with interpret=True on CPU
def _topk_topp_mask_and_sample_auto_interpret(*args, **kwargs):
  """Wrapper that automatically sets interpret=True on CPU."""
  if is_cpu_platform():
    kwargs['interpret'] = True
  return topk_topp_mask_and_sample(*args, **kwargs)


def reference_topk_topp_mask_and_sample(
  logits: jax.Array,
  rng_key: jax.Array,
  k: jax.Array,
  p: jax.Array,
  temperature: jax.Array,
  *,
  stable: bool = True,
  replace_val: float = -1e12,
) -> jax.Array:
  """Reference implementation using standalone vLLM functions.

  Args:
    logits: Input logits [batch, vocab_size]
    rng_key: RNG key
    k: Top-k values [batch] or scalar
    p: Top-p values [batch] or scalar
    temperature: Temperature values [batch] or scalar
    stable: Whether to use stable masking
    replace_val: Replacement value for masked elements

  Returns:
    Sampled token indices [batch]
  """
  batch_size = logits.shape[0]

  # Ensure correct shapes
  k = jnp.broadcast_to(k, (batch_size,))
  p = jnp.broadcast_to(p, (batch_size,))
  temperature = jnp.broadcast_to(temperature, (batch_size,))

  # Greedy sampling
  greedy_sampled = jnp.argmax(logits, axis=-1)

  # Apply logits to f32 for masking
  logits = logits.astype(jnp.float32)

  # Apply top-k masking
  logits_masked = jax.vmap(lambda l, k_val: topk_mask(l, k_val, replace_val=replace_val, stable=stable))(
    logits, k
  )

  # Apply top-p masking (stable=False for topp, only topk uses stable)
  logits_masked = jax.vmap(lambda l, p_val: topp_mask(l, p_val, replace_val=replace_val, stable=False))(
    logits_masked, p
  )

  # Apply temperature
  temperature_expanded = jnp.expand_dims(temperature, axis=-1)
  logits_masked = logits_masked / temperature_expanded.astype(logits_masked.dtype)

  # Sample
  next_tokens = jax.random.categorical(rng_key, logits_masked)

  # Use greedy when temperature is too low
  return jnp.where(temperature < _SAMPLING_EPS, greedy_sampled, next_tokens)


@pytest.mark.parametrize(
  "shape",
  [
    (8, 1024),
    (8, 2048),
    (8, 4096),
    (16, 2048),
    (24, 4096),
    (32, 8192),
    (13, 3570),  # Non-aligned batch size
  ],
)
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("stable", [True, False])
@pytest.mark.parametrize("seed", [42, 123, 456])
@pytest.mark.skipif(
  is_cpu_platform(),
  reason="Pallas tests require TPU/GPU - CPU uses interpret mode which is slow",
)
def test_topk_topp_mask_and_sample_vs_reference(shape, dtype, stable, seed):
  """Test topk_topp_mask_and_sample against reference implementation.

  Validates that the Pallas kernel exactly matches the reference implementation
  that uses jax.lax.top_k, reference topp_mask, and jax.random.categorical.
  """
  batch_size, vocab_size = shape

  # Split main seed into all needed keys
  key = jax.random.PRNGKey(seed)
  key, topk_key, topp_key, temp_key, logits_key, sample_key = jax.random.split(
    key, 6
  )

  # Create varying sampling parameters
  k = jax.random.randint(topk_key, (batch_size,), 1, 128, dtype=jnp.int32)
  p = jax.random.uniform(topp_key, (batch_size,), dtype=jnp.float32, minval=0.5, maxval=1.0)
  temperature = 10 ** jax.random.normal(temp_key, (batch_size,), dtype=jnp.float32)

  # Generate random logits
  logits = jax.random.normal(logits_key, shape).astype(dtype)

  # Run Pallas implementation
  pallas_result = _topk_topp_mask_and_sample_auto_interpret(
    logits,
    sample_key,
    k,
    p,
    temperature,
    stable=stable,
    block_token=8,
  )

  # Run reference implementation
  reference_result = reference_topk_topp_mask_and_sample(
    logits,
    sample_key,
    k,
    p,
    temperature,
    stable=stable,
  )

  # Compare results - expect exact match
  np.testing.assert_array_equal(
    pallas_result,
    reference_result,
    err_msg=f"Pallas kernel should exactly match reference for "
    f"shape={shape}, dtype={dtype}, stable={stable}, seed={seed}",
  )


@pytest.mark.parametrize(
  "shape",
  [
    (8, 1024),
    (16, 2048),
  ],
)
@pytest.mark.parametrize("dtype", [jnp.float32])
@pytest.mark.parametrize("seed", [42])
def test_topk_mask_correctness(shape, dtype, seed):
  """Test that top-k masking matches jax.lax.top_k behavior.

  Validates that after top-k masking, only the top-k values remain unmasked.
  """
  batch_size, vocab_size = shape

  key = jax.random.PRNGKey(seed)
  logits_key, topk_key, sample_key = jax.random.split(key, 3)

  # Create test data with a fixed k
  k = jax.random.randint(topk_key, (batch_size,), 10, 64, dtype=jnp.int32)
  p = jnp.ones((batch_size,), dtype=jnp.float32)  # No top-p filtering
  temperature = jnp.ones((batch_size,), dtype=jnp.float32)

  logits = jax.random.normal(logits_key, shape).astype(dtype)

  # Run the full kernel
  sampled = _topk_topp_mask_and_sample_auto_interpret(
    logits,
    sample_key,
    k,
    p,
    temperature,
    stable=True,
    block_token=8,
  )

  # Verify that sampled tokens are within top-k
  for i in range(batch_size):
    topk_values, topk_indices = jax.lax.top_k(logits[i], int(k[i]))
    assert sampled[i] in topk_indices, (
      f"Sampled token {sampled[i]} not in top-{k[i]} indices {topk_indices}"
    )


@pytest.mark.parametrize(
  "shape",
  [
    (8, 1024),
  ],
)
@pytest.mark.parametrize("seed", [42])
def test_greedy_sampling_low_temperature(shape, seed):
  """Test that greedy sampling is used when temperature is very low.

  When temperature < _SAMPLING_EPS, should return argmax instead of sampling.
  """
  batch_size, vocab_size = shape

  key = jax.random.PRNGKey(seed)
  logits_key, sample_key = jax.random.split(key, 2)

  # Create test data with very low temperature
  k = jnp.full((batch_size,), 64, dtype=jnp.int32)
  p = jnp.ones((batch_size,), dtype=jnp.float32)
  temperature = jnp.full((batch_size,), 1e-10, dtype=jnp.float32)  # Very low

  logits = jax.random.normal(logits_key, shape).astype(jnp.float32)

  # Run the kernel
  sampled = _topk_topp_mask_and_sample_auto_interpret(
    logits,
    sample_key,
    k,
    p,
    temperature,
    stable=True,
    block_token=8,
  )

  # Should match greedy (argmax)
  greedy = jnp.argmax(logits, axis=-1)
  np.testing.assert_array_equal(
    sampled,
    greedy,
    err_msg="Low temperature should trigger greedy sampling",
  )


@pytest.mark.parametrize(
  "batch_size",
  [7, 13, 17, 23],  # Test various non-aligned batch sizes
)
@pytest.mark.parametrize("seed", [42])
def test_padding_handling(batch_size, seed):
  """Test that padding is correctly handled for non-aligned batch sizes."""
  vocab_size = 1024

  key = jax.random.PRNGKey(seed)
  logits_key, sample_key = jax.random.split(key, 2)

  k = jnp.full((batch_size,), 32, dtype=jnp.int32)
  p = jnp.full((batch_size,), 0.9, dtype=jnp.float32)
  temperature = jnp.ones((batch_size,), dtype=jnp.float32)

  logits = jax.random.normal(logits_key, (batch_size, vocab_size)).astype(jnp.float32)

  # Run the kernel - should handle padding internally
  sampled = _topk_topp_mask_and_sample_auto_interpret(
    logits,
    sample_key,
    k,
    p,
    temperature,
    stable=True,
    block_token=8,
  )

  # Result should have correct batch size (no padding in output)
  assert sampled.shape == (batch_size,), f"Expected shape ({batch_size},), got {sampled.shape}"

  # All sampled tokens should be valid indices
  assert jnp.all((sampled >= 0) & (sampled < vocab_size)), "Sampled tokens out of bounds"


@pytest.mark.parametrize("seed", [42, 123])
def test_deterministic_sampling(seed):
  """Test that sampling is deterministic given the same RNG key."""
  batch_size, vocab_size = 16, 2048

  key = jax.random.PRNGKey(seed)
  logits_key, sample_key = jax.random.split(key, 2)

  k = jnp.full((batch_size,), 64, dtype=jnp.int32)
  p = jnp.full((batch_size,), 0.95, dtype=jnp.float32)
  temperature = jnp.ones((batch_size,), dtype=jnp.float32)

  logits = jax.random.normal(logits_key, (batch_size, vocab_size)).astype(jnp.float32)

  # Run twice with same key
  result1 = _topk_topp_mask_and_sample_auto_interpret(
    logits, sample_key, k, p, temperature, stable=True, block_token=8
  )
  result2 = _topk_topp_mask_and_sample_auto_interpret(
    logits, sample_key, k, p, temperature, stable=True, block_token=8
  )

  np.testing.assert_array_equal(
    result1,
    result2,
    err_msg="Same RNG key should produce identical samples",
  )


if __name__ == "__main__":
  print("Running topk_topp_mask_and_sample tests...")

  shapes = [(8, 1024), (16, 2048)]
  dtypes = [jnp.bfloat16, jnp.float32]
  stables = [True, False]
  seeds = [42, 123]

  print("\n=== Testing against reference implementation ===")
  for shape in shapes:
    for dtype in dtypes:
      for stable in stables:
        for seed in seeds:
          print(f"Testing shape={shape}, dtype={dtype}, stable={stable}, seed={seed}...", end=" ")
          test_topk_topp_mask_and_sample_vs_reference(shape, dtype, stable, seed)
          print("✓")

  print("\n=== Testing top-k mask correctness ===")
  for shape in [(8, 1024), (16, 2048)]:
    print(f"Testing shape={shape}...", end=" ")
    test_topk_mask_correctness(shape, jnp.float32, 42)
    print("✓")

  print("\n=== Testing greedy sampling ===")
  print("Testing low temperature...", end=" ")
  test_greedy_sampling_low_temperature((8, 1024), 42)
  print("✓")

  print("\n=== Testing padding handling ===")
  for batch_size in [7, 13, 17, 23]:
    print(f"Testing batch_size={batch_size}...", end=" ")
    test_padding_handling(batch_size, 42)
    print("✓")

  print("\n=== Testing deterministic sampling ===")
  for seed in [42, 123]:
    print(f"Testing seed={seed}...", end=" ")
    test_deterministic_sampling(seed)
    print("✓")

  print("\n✓ All tests passed!")
