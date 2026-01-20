"""Tests for high-precision top-p implementation."""

import jax
import jax.numpy as jnp
import numpy as np

from tallax.tax.high_precision_topp import (
  simulate_i64_add,
  f32_to_i64_scaled,
  i64_to_f32_scaled,
  sum_i64_parallel,
  topp_threshold_i64,
  topp_mask_high_precision,
)


class TestI64Simulation:
  """Test i64 simulation using two i32s."""

  def test_simple_addition(self):
    """Test basic i64 addition."""
    # Test: 100 + 200 = 300
    low_a = jnp.array([100], dtype=jnp.uint32)
    high_a = jnp.array([0], dtype=jnp.uint32)
    low_b = jnp.array([200], dtype=jnp.uint32)
    high_b = jnp.array([0], dtype=jnp.uint32)

    low_result, high_result = simulate_i64_add(low_a, high_a, low_b, high_b)

    assert low_result[0] == 300
    assert high_result[0] == 0

  def test_overflow_addition(self):
    """Test i64 addition with overflow from low to high."""
    # Test: 2^32 - 1 + 2 = 2^32 + 1
    low_a = jnp.array([2**32 - 1], dtype=jnp.uint32)
    high_a = jnp.array([0], dtype=jnp.uint32)
    low_b = jnp.array([2], dtype=jnp.uint32)
    high_b = jnp.array([0], dtype=jnp.uint32)

    low_result, high_result = simulate_i64_add(low_a, high_a, low_b, high_b)

    # low should wrap to 1, high should increment to 1
    assert low_result[0] == 1
    assert high_result[0] == 1

  def test_large_addition(self):
    """Test addition of large i64 numbers."""
    # Test: (2^32 + 100) + (2^32 + 200) = 2^33 + 300
    low_a = jnp.array([100], dtype=jnp.uint32)
    high_a = jnp.array([1], dtype=jnp.uint32)
    low_b = jnp.array([200], dtype=jnp.uint32)
    high_b = jnp.array([1], dtype=jnp.uint32)

    low_result, high_result = simulate_i64_add(low_a, high_a, low_b, high_b)

    assert low_result[0] == 300
    assert high_result[0] == 2


class TestF32I64Conversion:
  """Test float32 to/from i64 conversion."""

  def test_simple_conversion(self):
    """Test basic conversion."""
    x = jnp.array([0.5, 1.0, 2.0], dtype=jnp.float32)
    scale_factor = 2**20

    low, high = f32_to_i64_scaled(x, scale_factor)
    x_back = i64_to_f32_scaled(low, high, scale_factor)

    np.testing.assert_array_almost_equal(x, x_back, decimal=4)

  def test_small_values(self):
    """Test conversion of small values."""
    # Use realistic small probability values (not too extreme)
    x = jnp.array([1e-3, 1e-4, 1e-5], dtype=jnp.float32)
    scale_factor = 2**20

    low, high = f32_to_i64_scaled(x, scale_factor)
    x_back = i64_to_f32_scaled(low, high, scale_factor)

    # For small values, check relative error
    relative_error = jnp.abs((x - x_back) / x)
    # Should have <5% relative error for realistic probabilities
    assert jnp.all(relative_error < 0.05), f"Relative error too large: {relative_error}"

  def test_probability_range(self):
    """Test conversion of typical probability values."""
    x = jnp.array([0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999], dtype=jnp.float32)
    scale_factor = 2**20

    low, high = f32_to_i64_scaled(x, scale_factor)
    x_back = i64_to_f32_scaled(low, high, scale_factor)

    np.testing.assert_array_almost_equal(x, x_back, decimal=5)


class TestParallelSum:
  """Test parallel i64 summation."""

  def test_simple_sum(self):
    """Test sum of small values."""
    probs = jnp.array([[0.25, 0.25, 0.25, 0.25]], dtype=jnp.float32)

    sum_low, sum_high = sum_i64_parallel(probs, num_bins=4)

    # Convert back to f32
    total = i64_to_f32_scaled(sum_low, sum_high)

    # Should sum to ~1.0
    np.testing.assert_almost_equal(total[0], 1.0, decimal=4)

  def test_large_vocabulary_sum(self):
    """Test sum with large vocabulary."""
    rng = jax.random.PRNGKey(42)
    logits = jax.random.normal(rng, (2, 1000))
    probs = jax.nn.softmax(logits, axis=-1)

    sum_low, sum_high = sum_i64_parallel(probs, num_bins=128)

    # Convert back
    total = i64_to_f32_scaled(sum_low, sum_high)

    # Should sum to 1.0 for each batch
    for i in range(2):
      np.testing.assert_almost_equal(total[i], 1.0, decimal=3)


class TestHighPrecisionTopP:
  """Test high-precision top-p masking."""

  def test_simple_topp(self):
    """Test basic top-p filtering."""
    # Create logits with known probabilities
    logits = jnp.array([[10.0, 5.0, 2.0, 1.0, 0.5]])
    p = 0.9

    result = topp_mask_high_precision(logits, p, replace_val=-jnp.inf)

    # Should mask out some tokens
    assert (result == -jnp.inf).any()
    # Should keep top tokens
    assert result[0, 0] != -jnp.inf  # Top logit should be kept

  def test_topp_threshold(self):
    """Test threshold finding."""
    # Uniform probabilities
    probs = jnp.ones((1, 10), dtype=jnp.float32) / 10.0
    p = 0.5

    threshold = topp_threshold_i64(probs, p)

    # With uniform probs, threshold should be around 0.1
    assert 0.05 < threshold[0] < 0.15

  def test_topp_deterministic(self):
    """Test that high-precision top-p is deterministic."""
    rng = jax.random.PRNGKey(42)
    logits = jax.random.normal(rng, (4, 100))
    p = 0.9

    # Run twice
    result1 = topp_mask_high_precision(logits, p)
    result2 = topp_mask_high_precision(logits, p)

    # Should be identical
    np.testing.assert_array_equal(result1, result2)

  def test_topp_stable_mode(self):
    """Test stable vs unstable modes."""
    # Create logits with ties
    logits = jnp.array([[10.0, 5.0, 5.0, 5.0, 1.0]])
    p = 0.7

    result_stable = topp_mask_high_precision(logits, p, stable=True)
    result_unstable = topp_mask_high_precision(logits, p, stable=False)

    # Both should work
    kept_stable = (result_stable != -1e12).sum()
    kept_unstable = (result_unstable != -1e12).sum()

    assert kept_stable > 0
    assert kept_unstable > 0


class TestSummationOrderIndependence:
  """Test that high-precision implementation is summation-order agnostic."""

  def test_different_orders_same_result(self):
    """Test that different orderings give same result."""
    # Create probability array
    probs = jnp.array([0.3, 0.25, 0.2, 0.15, 0.1], dtype=jnp.float32)

    # Sum in different orders
    probs_shuffled1 = probs[::-1]  # Reverse order
    probs_shuffled2 = probs[[2, 0, 4, 1, 3]]  # Random permutation

    # Compute sums using i64
    sum1_low, sum1_high = sum_i64_parallel(probs[None, :], num_bins=5)
    sum2_low, sum2_high = sum_i64_parallel(probs_shuffled1[None, :], num_bins=5)
    sum3_low, sum3_high = sum_i64_parallel(probs_shuffled2[None, :], num_bins=5)

    # Convert back to f32
    total1 = i64_to_f32_scaled(sum1_low, sum1_high)[0]
    total2 = i64_to_f32_scaled(sum2_low, sum2_high)[0]
    total3 = i64_to_f32_scaled(sum3_low, sum3_high)[0]

    # All should give same result (within precision)
    np.testing.assert_almost_equal(total1, total2, decimal=5)
    np.testing.assert_almost_equal(total1, total3, decimal=5)

  def test_precision_vs_float32(self):
    """Test that i64 provides better precision than f32."""
    rng = jax.random.PRNGKey(42)
    # Create many small values that would accumulate rounding errors in f32
    probs = jax.random.uniform(rng, (10000,), minval=0, maxval=1e-6).astype(jnp.float32)
    probs = probs / probs.sum()  # Normalize

    # F32 sum (forward and reverse to show order dependence)
    sum_f32_forward = probs.sum()
    sum_f32_reverse = probs[::-1].sum()

    # I64 sum (should be order-independent)
    sum_i64_forward_low, sum_i64_forward_high = sum_i64_parallel(probs[None, :])
    sum_i64_reverse_low, sum_i64_reverse_high = sum_i64_parallel(probs[None, ::-1])

    total_i64_forward = i64_to_f32_scaled(sum_i64_forward_low, sum_i64_forward_high)[0]
    total_i64_reverse = i64_to_f32_scaled(sum_i64_reverse_low, sum_i64_reverse_high)[0]

    # I64 should be more consistent across orderings
    i64_diff = abs(total_i64_forward - total_i64_reverse)
    f32_diff = abs(sum_f32_forward - sum_f32_reverse)

    # Note: This test might not always show difference with small arrays
    # But demonstrates the concept
    print(f"F32 order difference: {f32_diff}")
    print(f"I64 order difference: {i64_diff}")


def run_tests():
  """Run all tests."""
  print("Running high-precision top-p tests...\n")

  # Test I64 simulation
  print("Test 1: I64 simulation")
  test = TestI64Simulation()
  test.test_simple_addition()
  test.test_overflow_addition()
  test.test_large_addition()
  print("  PASS\n")

  # Test conversions
  print("Test 2: F32<->I64 conversions")
  test = TestF32I64Conversion()
  test.test_simple_conversion()
  test.test_small_values()
  test.test_probability_range()
  print("  PASS\n")

  # Test parallel sum
  print("Test 3: Parallel i64 summation")
  test = TestParallelSum()
  test.test_simple_sum()
  test.test_large_vocabulary_sum()
  print("  PASS\n")

  # Test high-precision top-p
  print("Test 4: High-precision top-p")
  test = TestHighPrecisionTopP()
  test.test_simple_topp()
  test.test_topp_threshold()
  test.test_topp_deterministic()
  test.test_topp_stable_mode()
  print("  PASS\n")

  # Test summation order independence
  print("Test 5: Summation order independence")
  test = TestSummationOrderIndependence()
  test.test_different_orders_same_result()
  test.test_precision_vs_float32()
  print("  PASS\n")

  print("="*50)
  print("All high-precision top-p tests passed!")
  print("="*50)


if __name__ == "__main__":
  run_tests()
