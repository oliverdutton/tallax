"""Tests for individual sampling components against the reference implementation.

Tests that each substage of both the fullvocab and reducedk kernels
matches the (possibly transformed) reference computation.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from tallax.vllm.reference import reference_topk_topp_mask_and_sample
from tallax.vllm.fullvocab.kernel import topk_topp_mask_and_sample as fullvocab_sample
from tallax.vllm.fullvocab.topk_mask import topk_mask_pallas
from tallax.vllm.fullvocab.topp_mask import topp_mask
from tallax.vllm.utils.high_precision_uint import (
  U48,
  modulo_u128_u64,
  sample_random_u128_in_u32s,
)
from tallax.vllm.utils.binary_search import (
  binary_search,
  monotonic_f32_to_u32,
  monotonic_u32_to_f32,
)
from tallax.tax.utils import is_cpu_platform


# ---------------------------------------------------------------------------
# Utility: shared test parameters
# ---------------------------------------------------------------------------

SEEDS = [42, 123]
BATCH_SIZES = [1, 8]
VOCAB_SIZES = [256, 1024]


def _make_inputs(seed, batch_size, vocab_size):
  key = jax.random.PRNGKey(seed)
  keys = jax.random.split(key, 6)
  logits = jax.random.normal(keys[0], (batch_size, vocab_size), dtype=jnp.float32)
  k = jax.random.randint(keys[1], (batch_size,), 1, min(64, vocab_size), dtype=jnp.int32)
  p = jax.random.uniform(keys[2], (batch_size,), dtype=jnp.float32, minval=0.1, maxval=1.0)
  temperature = 10 ** jax.random.normal(keys[3], (batch_size,), dtype=jnp.float32)
  temperature = jnp.clip(temperature, 0.1, 10.0)
  rng_key = keys[4]
  return logits, k, p, temperature, rng_key


# ---------------------------------------------------------------------------
# Test: monotonic f32 <-> u32 roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_monotonic_f32_u32_roundtrip(seed):
  """monotonic_f32_to_u32 and monotonic_u32_to_f32 are inverses."""
  key = jax.random.PRNGKey(seed)
  vals = jax.random.normal(key, (100,), dtype=jnp.float32)
  roundtripped = monotonic_u32_to_f32(monotonic_f32_to_u32(vals))
  np.testing.assert_array_equal(vals, roundtripped)


@pytest.mark.parametrize("seed", SEEDS)
def test_monotonic_preserves_order(seed):
  """monotonic_f32_to_u32 preserves ordering."""
  key = jax.random.PRNGKey(seed)
  vals = jax.random.normal(key, (100,), dtype=jnp.float32)
  sorted_vals = jnp.sort(vals)
  sorted_u32 = monotonic_f32_to_u32(sorted_vals)
  # u32 values should also be sorted
  assert jnp.all(sorted_u32[1:] >= sorted_u32[:-1])


# ---------------------------------------------------------------------------
# Test: U48 arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_u48_sum_matches_i64(seed):
  """U48.map_reduce_sum matches direct i64 sum."""
  key = jax.random.PRNGKey(seed)
  scale = 2**24 - 1
  vals = jax.random.randint(key, (4, 512), 0, scale, dtype=jnp.int32)
  u48_sum = U48.map_reduce_sum(vals, max_val=scale)
  u48_as_f32 = u48_sum.to_f32()

  with jax.enable_x64(True):
    expected = vals.astype(jnp.int64).sum(axis=1, keepdims=True).astype(jnp.float64)

  np.testing.assert_allclose(
    u48_as_f32.astype(float),
    np.array(expected).astype(float),
    rtol=1e-6,
    err_msg="U48 sum should match i64 sum",
  )


@pytest.mark.parametrize("seed", SEEDS)
def test_u48_comparison(seed):
  """U48 < operator is consistent with f32 comparison."""
  key = jax.random.PRNGKey(seed)
  a_vals = jax.random.randint(key, (10,), 0, 2**24 - 1, dtype=jnp.int32)
  key = jax.random.split(key)[0]
  b_vals = jax.random.randint(key, (10,), 0, 2**24 - 1, dtype=jnp.int32)
  a = U48(a_vals, max_val=2**24 - 1)
  b = U48(b_vals, max_val=2**24 - 1)
  result = a < b
  expected = a_vals < b_vals
  np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Test: modulo_u128_u64
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_modulo_u128_u64(seed):
  """modulo_u128_u64 matches Python arbitrary precision."""
  key = jax.random.PRNGKey(seed)
  dividend = tuple(jax.random.bits(key, (4, 2, 1), jnp.uint32))
  key = jax.random.split(key)[0]
  # Use small divisor to test
  divisor_low = jax.random.randint(key, (2, 1), 1, 2**31, dtype=jnp.int32).astype(jnp.uint32)
  divisor = [jnp.zeros_like(divisor_low), divisor_low]

  result_h, result_l = modulo_u128_u64(dividend, divisor)

  # Verify with Python
  d = [np.array(x, dtype=object) for x in dividend]
  val_128 = (d[0] << 96) | (d[1] << 64) | (d[2] << 32) | d[3]
  m = np.array(divisor_low, dtype=object)
  expected = (val_128 % m).astype(np.uint64)
  actual = (np.array(result_h, dtype=np.uint64) << 32) + np.array(result_l, dtype=np.uint64)
  np.testing.assert_array_equal(
    actual, expected, err_msg="modulo_u128_u64 should match Python"
  )


# ---------------------------------------------------------------------------
# Test: binary_search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", [0.0, 0.5, -1.5, 3.14])
def test_binary_search_finds_target(target):
  """Binary search converges to the correct threshold."""
  target_arr = jnp.array([[target]], dtype=jnp.float32)

  def predicate(pivot):
    return pivot < target_arr

  lo = jnp.full((1, 1), -100.0, jnp.float32)
  hi = jnp.full((1, 1), 100.0, jnp.float32)
  l, threshold, _ = binary_search(predicate, lo, hi, num_iter=32)
  np.testing.assert_allclose(
    float(threshold), target, atol=1e-6,
    err_msg=f"Binary search should find target={target}",
  )


# ---------------------------------------------------------------------------
# Test: fullvocab topk_mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("k_val", [1, 10, 50])
@pytest.mark.skipif(
  is_cpu_platform(),
  reason="topk_mask_pallas requires TPU/GPU",
)
def test_topk_mask_count(seed, batch_size, k_val):
  """topk_mask keeps exactly k non-replaced values when stable=True."""
  key = jax.random.PRNGKey(seed)
  vocab_size = 256
  logits = jax.random.normal(key, (batch_size, vocab_size), dtype=jnp.float32)
  replace_val = -1e12
  k = jnp.full((batch_size,), k_val, dtype=jnp.int32)
  masked = topk_mask_pallas(logits, k, replace_val=replace_val, stable=True)
  counts = (masked != replace_val).sum(axis=1)
  np.testing.assert_array_equal(
    counts,
    jnp.full((batch_size,), k_val),
    err_msg=f"topk_mask should keep exactly k={k_val} elements",
  )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("k_val", [1, 10])
@pytest.mark.skipif(
  is_cpu_platform(),
  reason="topk_mask_pallas requires TPU/GPU",
)
def test_topk_mask_values(seed, batch_size, k_val):
  """topk_mask keeps the correct top-k values."""
  key = jax.random.PRNGKey(seed)
  vocab_size = 256
  logits = jax.random.normal(key, (batch_size, vocab_size), dtype=jnp.float32)
  replace_val = -1e12
  k = jnp.full((batch_size,), k_val, dtype=jnp.int32)
  masked = topk_mask_pallas(logits, k, replace_val=replace_val, stable=True)

  # Reference: jax.lax.top_k
  for b in range(batch_size):
    ref_vals, _ = jax.lax.top_k(logits[b], k_val)
    actual_vals = jnp.sort(masked[b][masked[b] != replace_val])[::-1]
    np.testing.assert_allclose(
      actual_vals[:k_val], ref_vals, atol=1e-5,
      err_msg=f"topk_mask values should match jax.lax.top_k for batch={b}",
    )


# ---------------------------------------------------------------------------
# Test: fullvocab topp_mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("p_val", [0.1, 0.5, 0.9, 1.0])
def test_topp_mask_nonzero_probability(seed, p_val):
  """topp_mask returns at least one nonzero token per batch."""
  key = jax.random.PRNGKey(seed)
  batch_size, vocab_size = 4, 256
  logits = jax.random.normal(key, (batch_size, vocab_size), dtype=jnp.float32)
  p = jnp.full((batch_size, 1), p_val, dtype=jnp.float32)
  result = topp_mask(logits, p)
  nonzero_count = (result != 0).sum(axis=1)
  assert jnp.all(nonzero_count > 0), (
    f"topp_mask should keep at least 1 token, got {nonzero_count}"
  )


# ---------------------------------------------------------------------------
# Test: reference implementation sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_reference_greedy(seed):
  """Reference with temperature ~0 should return argmax."""
  key = jax.random.PRNGKey(seed)
  batch_size, vocab_size = 4, 128
  logits = jax.random.normal(key, (batch_size, vocab_size), dtype=jnp.float32)
  k = jnp.full((batch_size,), vocab_size, dtype=jnp.int32)
  p = jnp.ones((batch_size,), dtype=jnp.float32)
  temperature = jnp.full((batch_size,), 1e-7, dtype=jnp.float32)
  rng_key = jax.random.split(key)[0]

  result = reference_topk_topp_mask_and_sample(
    logits, rng_key, k, p, temperature
  )
  expected = logits.argmax(axis=1)
  np.testing.assert_array_equal(
    result, expected,
    err_msg="Reference greedy sampling should return argmax",
  )


@pytest.mark.parametrize("seed", SEEDS)
def test_reference_debug_intermediates(seed):
  """Reference debug mode returns all expected intermediate keys."""
  key = jax.random.PRNGKey(seed)
  batch_size, vocab_size = 2, 128
  logits = jax.random.normal(key, (batch_size, vocab_size), dtype=jnp.float32)
  k = jnp.full((batch_size,), 10, dtype=jnp.int32)
  p = jnp.full((batch_size,), 0.9, dtype=jnp.float32)
  temperature = jnp.ones((batch_size,), dtype=jnp.float32)
  rng_key = jax.random.split(key)[0]

  result, debug = reference_topk_topp_mask_and_sample(
    logits, rng_key, k, p, temperature, debug=True
  )
  expected_keys = {
    "greedy_sampled",
    "topk_logits_unsorted",
    "topp_unnorm_probs_i32",
    "topp_nonzero_count",
    "total_sum",
    "next_tokens",
  }
  assert set(debug.keys()) == expected_keys, (
    f"Debug keys mismatch: {set(debug.keys())} != {expected_keys}"
  )
  # Greedy should be argmax
  np.testing.assert_array_equal(
    debug["greedy_sampled"], logits.argmax(axis=1)
  )


# ---------------------------------------------------------------------------
# Test: fullvocab vs reference end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("vocab_size", [256])
@pytest.mark.skipif(
  is_cpu_platform(),
  reason="fullvocab kernel requires TPU/GPU",
)
def test_fullvocab_matches_reference(seed, batch_size, vocab_size):
  """Full-vocabulary kernel should produce the same tokens as reference."""
  logits, k, p, temperature, rng_key = _make_inputs(seed, batch_size, vocab_size)

  ref_result = reference_topk_topp_mask_and_sample(
    logits, rng_key, k, p, temperature
  )
  fullvocab_result = fullvocab_sample(
    logits, rng_key, k, p, temperature, stable=True
  )
  np.testing.assert_array_equal(
    fullvocab_result, ref_result,
    err_msg=f"fullvocab should match reference for seed={seed}, batch={batch_size}",
  )


# ---------------------------------------------------------------------------
# Test: fullvocab debug mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(
  is_cpu_platform(),
  reason="fullvocab kernel requires TPU/GPU",
)
def test_fullvocab_debug_intermediates(seed):
  """fullvocab debug mode returns debug dict with expected keys."""
  logits, k, p, temperature, rng_key = _make_inputs(seed, 2, 256)

  result, debug = fullvocab_sample(
    logits, rng_key, k, p, temperature, stable=True, debug=True
  )
  ref_result = reference_topk_topp_mask_and_sample(
    logits, rng_key, k, p, temperature
  )
  # Results should still match
  np.testing.assert_array_equal(result, ref_result)
  # Debug dict should have the expected keys
  assert "greedy_sampled" in debug
  assert "next_tokens" in debug


if __name__ == "__main__":
  print("Running sampling component tests...")
  # Run a subset that doesn't require TPU
  for seed in SEEDS:
    test_monotonic_f32_u32_roundtrip(seed)
    test_monotonic_preserves_order(seed)
    test_u48_sum_matches_i64(seed)
    test_u48_comparison(seed)
    test_modulo_u128_u64(seed)
    test_reference_greedy(seed)
    test_reference_debug_intermediates(seed)
  for target in [0.0, 0.5, -1.5, 3.14]:
    test_binary_search_finds_target(target)
  print("All non-TPU component tests passed!")
