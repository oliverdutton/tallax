"""Tests for optimized topk_mask implementation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tallax.tax.optimized_topk_mask import (
  monotonic_f32_to_u32,
  monotonic_u32_to_f32,
  interp_f32,
  find_topk_threshold_jax,
  topk_mask_stable,
  stable_topk_mask_jax,
)


class TestMonotonicConversions:
  """Test monotonic f32<->u32 conversions."""

  def test_roundtrip(self):
    """Test that conversions roundtrip correctly."""
    # Test various float values
    test_values = jnp.array([
      -1e10, -100.0, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 100.0, 1e10
    ], dtype=jnp.float32)

    # Roundtrip conversion
    u32_values = monotonic_f32_to_u32(test_values)
    f32_back = monotonic_u32_to_f32(u32_values)

    np.testing.assert_array_equal(test_values, f32_back)

  def test_monotonicity(self):
    """Test that larger floats map to larger uint32 values."""
    # Create sorted float array
    test_values = jnp.array(
      [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0],
      dtype=jnp.float32
    )

    u32_values = monotonic_f32_to_u32(test_values)

    # Check that u32 values are also sorted
    assert jnp.all(u32_values[:-1] <= u32_values[1:])

  def test_interp_midpoint(self):
    """Test that interp_f32 produces reasonable midpoints."""
    l = jnp.array([0.0], dtype=jnp.float32)
    r = jnp.array([100.0], dtype=jnp.float32)

    m = interp_f32(l, r)

    # Midpoint should be between l and r
    assert jnp.all(m > l)
    assert jnp.all(m < r)

    # Should be approximately in the middle (in u32 space, not f32 space)
    # So it won't necessarily be 50.0


class TestBinarySearchThreshold:
  """Test binary search threshold finding."""

  def test_simple_case(self):
    """Test threshold finding on simple sorted array."""
    # Create array where we know the threshold
    x = jnp.array([[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    k = 3

    threshold = find_topk_threshold_jax(x, k)

    # Threshold should be 8.0 (the 3rd largest value)
    # Actually, threshold will be such that >= k values are >= threshold
    # So threshold should be <= 8.0 and > 7.0
    count_ge = (x >= threshold).sum()
    assert count_ge >= k

  def test_with_ties(self):
    """Test threshold finding with tied values."""
    x = jnp.array([[10.0, 9.0, 8.0, 8.0, 8.0, 5.0, 4.0, 3.0]])
    k = 4

    threshold = find_topk_threshold_jax(x, k)

    # Should find threshold at or just below 8.0
    count_ge = (x >= threshold).sum()
    assert count_ge >= k

  def test_batched(self):
    """Test with batched input."""
    x = jnp.array([
      [10.0, 9.0, 8.0, 7.0, 6.0],
      [5.0, 4.0, 3.0, 2.0, 1.0],
    ])
    k = 2

    threshold = find_topk_threshold_jax(x, k)

    # Check that each batch has >= k values >= threshold
    for i in range(2):
      count_ge = (x[i] >= threshold[i]).sum()
      assert count_ge >= k


class TestStableTopkMask:
  """Test stable topk_mask implementation."""

  def test_no_ties(self):
    """Test when there are no tied values."""
    x = jnp.array([[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    k = 5
    replace_val = -1e12

    result = topk_mask_stable(x, k, replace_val, stable=True)

    # Should keep exactly top 5 values: 10, 9, 8, 7, 6
    expected = jnp.array([[10.0, 9.0, 8.0, 7.0, 6.0, -1e12, -1e12, -1e12, -1e12, -1e12]])

    np.testing.assert_array_almost_equal(result, expected)

  def test_with_ties_stable(self):
    """Test stable behavior with tied values."""
    # Array with ties at boundary
    x = jnp.array([[10.0, 9.0, 8.0, 8.0, 8.0, 7.0, 6.0, 5.0]])
    k = 5

    result = topk_mask_stable(x, k, -1e12, stable=True)

    # Count how many elements are kept (not -1e12)
    kept = (result != -1e12).sum()

    # Should keep exactly k elements
    # Due to stable sorting, should keep first k elements that are >= threshold
    # The threshold will be 8.0, and we keep first 5 elements with value >= 8.0
    # That's indices 0, 1, 2, 3, 4 (values 10, 9, 8, 8, 8)
    assert kept == k

  def test_compare_with_jax_topk(self):
    """Compare stable version with jax.lax.top_k."""
    # Create test array
    rng = jax.random.PRNGKey(42)
    x = jax.random.uniform(rng, (4, 100), minval=-10, maxval=10)
    k = 10

    # Our implementation
    our_result = topk_mask_stable(x, k, -jnp.inf, stable=True)
    our_topk_vals = jnp.sort(our_result[our_result != -jnp.inf].reshape(4, -1), axis=-1)[:, ::-1]

    # JAX implementation
    jax_topk_vals, _ = jax.lax.top_k(x, k)

    # Values should match (order might differ for ties, but sorted values should match)
    np.testing.assert_array_almost_equal(
      jnp.sort(our_topk_vals, axis=-1)[:, ::-1],
      jnp.sort(jax_topk_vals, axis=-1)[:, ::-1],
      decimal=5
    )

  def test_unstable_allows_more_than_k(self):
    """Test that unstable version can return more than k elements when tied."""
    x = jnp.array([[10.0, 8.0, 8.0, 8.0, 8.0, 8.0, 5.0, 3.0]])
    k = 3

    result = topk_mask_stable(x, k, -1e12, stable=False)

    # Unstable version should keep all values >= threshold
    # threshold will be 8.0, so it keeps 1 value of 10 and 5 values of 8 = 6 total
    kept = (result != -1e12).sum()

    # Should keep more than k elements
    assert kept > k


class TestEdgeCases:
  """Test edge cases."""

  def test_k_equals_vocab_size(self):
    """Test when k equals vocabulary size."""
    x = jnp.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
    k = 5

    result = topk_mask_stable(x, k, -1e12, stable=True)

    # Should keep all elements
    np.testing.assert_array_equal(result, x)

  def test_k_equals_one(self):
    """Test when k=1."""
    x = jnp.array([[10.0, 9.0, 8.0, 7.0, 6.0]])
    k = 1

    result = topk_mask_stable(x, k, -1e12, stable=True)

    # Should keep only the maximum value
    kept = (result != -1e12).sum()
    assert kept == k
    assert result[0, 0] == 10.0

  def test_all_same_values(self):
    """Test when all values are the same."""
    x = jnp.array([[5.0, 5.0, 5.0, 5.0, 5.0]])
    k = 3

    result = topk_mask_stable(x, k, -1e12, stable=True)

    # Should keep exactly k elements (stable version)
    kept = (result != -1e12).sum()
    assert kept == k


if __name__ == "__main__":
  # Run tests
  pytest.main([__file__, "-v"])
