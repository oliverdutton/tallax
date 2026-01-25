"""Simple test for topk_topp_mask_and_sample Pallas kernel without pytest.

This test validates the combined top-k, top-p masking and sampling kernel against:
1. jax.lax.top_k for top-k masking (f32 and stable)
2. Reference implementation from tpu_inference_sampling for top-p
3. jax.random.categorical for sampling
4. Exact match on sampled tokens
"""

import jax
import jax.numpy as jnp
import numpy as np

from tallax.vllm.topk_topp_mask_and_sample import topk_topp_mask_and_sample
from tallax.vllm.tpu_inference_sampling_as_standalone_file import (
  topk_mask,
  topp_mask,
  _SAMPLING_EPS,
)


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


def test_basic_functionality():
  """Test basic functionality with small inputs."""
  print("Testing basic functionality...", end=" ")

  batch_size, vocab_size = 8, 1024
  seed = 42

  key = jax.random.PRNGKey(seed)
  logits_key, sample_key = jax.random.split(key, 2)

  k = jnp.full((batch_size,), 32, dtype=jnp.int32)
  p = jnp.full((batch_size,), 0.9, dtype=jnp.float32)
  temperature = jnp.ones((batch_size,), dtype=jnp.float32)

  logits = jax.random.normal(logits_key, (batch_size, vocab_size)).astype(jnp.float32)

  # Run Pallas implementation
  result = topk_topp_mask_and_sample(
    logits,
    sample_key,
    k,
    p,
    temperature,
    stable=True,
    block_token=8,
    interpret=True,  # Use interpret mode for CPU
  )

  # Check output shape and validity
  assert result.shape == (batch_size,), f"Expected shape ({batch_size},), got {result.shape}"
  assert jnp.all((result >= 0) & (result < vocab_size)), "Sampled tokens out of bounds"

  print("✓")


def test_vs_reference():
  """Test against reference implementation."""
  print("Testing against reference implementation...", end=" ")

  batch_size, vocab_size = 8, 1024
  seed = 42

  key = jax.random.PRNGKey(seed)
  key, topk_key, topp_key, temp_key, logits_key, sample_key = jax.random.split(key, 6)

  # Create varying sampling parameters
  k = jax.random.randint(topk_key, (batch_size,), 1, 64, dtype=jnp.int32)
  p = jax.random.uniform(topp_key, (batch_size,), dtype=jnp.float32, minval=0.5, maxval=1.0)
  temperature = jnp.ones((batch_size,), dtype=jnp.float32)

  # Generate random logits
  logits = jax.random.normal(logits_key, (batch_size, vocab_size)).astype(jnp.float32)

  # Run Pallas implementation
  pallas_result = topk_topp_mask_and_sample(
    logits,
    sample_key,
    k,
    p,
    temperature,
    stable=True,
    block_token=8,
    interpret=True,  # Use interpret mode for CPU
  )

  # Run reference implementation
  reference_result = reference_topk_topp_mask_and_sample(
    logits,
    sample_key,
    k,
    p,
    temperature,
    stable=True,
  )

  # Compare results
  if not jnp.array_equal(pallas_result, reference_result):
    print("\nMismatch found!")
    print(f"Pallas result: {pallas_result}")
    print(f"Reference result: {reference_result}")
    print(f"Differences at indices: {jnp.where(pallas_result != reference_result)[0]}")
    raise AssertionError("Results don't match")

  print("✓")


def test_greedy_sampling():
  """Test greedy sampling with low temperature."""
  print("Testing greedy sampling...", end=" ")

  batch_size, vocab_size = 8, 1024
  seed = 42

  key = jax.random.PRNGKey(seed)
  logits_key, sample_key = jax.random.split(key, 2)

  k = jnp.full((batch_size,), 64, dtype=jnp.int32)
  p = jnp.ones((batch_size,), dtype=jnp.float32)
  temperature = jnp.full((batch_size,), 1e-10, dtype=jnp.float32)

  logits = jax.random.normal(logits_key, (batch_size, vocab_size)).astype(jnp.float32)

  # Run the kernel
  sampled = topk_topp_mask_and_sample(
    logits,
    sample_key,
    k,
    p,
    temperature,
    stable=True,
    block_token=8,
    interpret=True,
  )

  # Should match greedy (argmax)
  greedy = jnp.argmax(logits, axis=-1)
  assert jnp.array_equal(sampled, greedy), "Low temperature should trigger greedy sampling"

  print("✓")


def test_padding():
  """Test padding handling for non-aligned batch sizes."""
  print("Testing padding handling...", end=" ")

  batch_sizes = [7, 13, 17, 23]
  vocab_size = 1024
  seed = 42

  for batch_size in batch_sizes:
    key = jax.random.PRNGKey(seed)
    logits_key, sample_key = jax.random.split(key, 2)

    k = jnp.full((batch_size,), 32, dtype=jnp.int32)
    p = jnp.full((batch_size,), 0.9, dtype=jnp.float32)
    temperature = jnp.ones((batch_size,), dtype=jnp.float32)

    logits = jax.random.normal(logits_key, (batch_size, vocab_size)).astype(jnp.float32)

    # Run the kernel
    sampled = topk_topp_mask_and_sample(
      logits,
      sample_key,
      k,
      p,
      temperature,
      stable=True,
      block_token=8,
      interpret=True,
    )

    assert sampled.shape == (batch_size,), f"Expected shape ({batch_size},), got {sampled.shape}"
    assert jnp.all((sampled >= 0) & (sampled < vocab_size)), "Sampled tokens out of bounds"

  print("✓")


def test_deterministic():
  """Test deterministic sampling."""
  print("Testing deterministic sampling...", end=" ")

  batch_size, vocab_size = 16, 2048
  seed = 42

  key = jax.random.PRNGKey(seed)
  logits_key, sample_key = jax.random.split(key, 2)

  k = jnp.full((batch_size,), 64, dtype=jnp.int32)
  p = jnp.full((batch_size,), 0.95, dtype=jnp.float32)
  temperature = jnp.ones((batch_size,), dtype=jnp.float32)

  logits = jax.random.normal(logits_key, (batch_size, vocab_size)).astype(jnp.float32)

  # Run twice with same key
  result1 = topk_topp_mask_and_sample(
    logits, sample_key, k, p, temperature, stable=True, block_token=8, interpret=True
  )
  result2 = topk_topp_mask_and_sample(
    logits, sample_key, k, p, temperature, stable=True, block_token=8, interpret=True
  )

  assert jnp.array_equal(result1, result2), "Same RNG key should produce identical samples"

  print("✓")


if __name__ == "__main__":
  print("\n" + "="*60)
  print("Running topk_topp_mask_and_sample tests")
  print("="*60 + "\n")

  try:
    test_basic_functionality()
    test_greedy_sampling()
    test_padding()
    test_deterministic()
    test_vs_reference()

    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
  except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
