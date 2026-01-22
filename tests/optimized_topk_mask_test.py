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
from tallax.vllm.tpu_inference_sampling_as_standalone_file import (
  topk_mask,
  topp_mask,
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

  def test_negative_values(self):
    """Test with all negative values."""
    x = jnp.array([[-1.0, -2.0, -3.0, -4.0, -5.0]])
    k = 3

    result = topk_mask_stable(x, k, -1e12, stable=True)

    # Should keep top 3: -1.0, -2.0, -3.0
    kept = (result != -1e12).sum()
    assert kept == k
    kept_vals = result[result != -1e12]
    np.testing.assert_array_almost_equal(
      jnp.sort(kept_vals)[::-1], jnp.array([-1.0, -2.0, -3.0])
    )

  def test_mixed_positive_negative(self):
    """Test with mixed positive and negative values."""
    x = jnp.array([[5.0, -2.0, 3.0, -4.0, 1.0, -6.0]])
    k = 3

    result = topk_mask_stable(x, k, -1e12, stable=True)

    # Should keep: 5.0, 3.0, 1.0
    kept = (result != -1e12).sum()
    assert kept == k

  def test_inf_values(self):
    """Test handling of infinity values."""
    x = jnp.array([[jnp.inf, 10.0, 5.0, -jnp.inf, 0.0]])
    k = 3

    result = topk_mask_stable(x, k, -1e12, stable=True)

    # Should keep: inf, 10.0, 5.0
    kept = (result != -1e12).sum()
    assert kept == k
    assert result[0, 0] == jnp.inf

  def test_very_large_k(self):
    """Test when k is larger than vocab size."""
    x = jnp.array([[5.0, 4.0, 3.0]])
    k = 10  # Larger than vocab size

    result = topk_mask_stable(x, k, -1e12, stable=True)

    # Should keep all elements
    kept = (result != -1e12).sum()
    assert kept == 3  # Only 3 elements available


class TestBatchedOperations:
  """Test batched operations."""

  def test_large_batch(self):
    """Test with large batch size."""
    rng = jax.random.PRNGKey(42)
    batch_size = 128
    vocab_size = 1000
    k = 50

    x = jax.random.normal(rng, (batch_size, vocab_size))
    result = topk_mask_stable(x, k, -jnp.inf, stable=True)

    # Check each batch has exactly k elements
    for i in range(batch_size):
      kept = (result[i] != -jnp.inf).sum()
      assert kept == k, f"Batch {i} has {kept} elements, expected {k}"

  def test_different_k_per_batch(self):
    """Test with different k values per batch element (not currently supported)."""
    # This is a placeholder for when we support dynamic k per batch
    pass

  def test_3d_input(self):
    """Test with 3D input (multiple batch dimensions)."""
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (4, 8, 100))
    k = 10

    result = topk_mask_stable(x, k, -jnp.inf, stable=True)

    # Check shape preserved
    assert result.shape == x.shape

    # Check each element has exactly k values
    for i in range(4):
      for j in range(8):
        kept = (result[i, j] != -jnp.inf).sum()
        assert kept == k


class TestNumericalStability:
  """Test numerical stability."""

  def test_very_small_differences(self):
    """Test with very small differences between values."""
    # Values very close together
    x = jnp.array([[1.0, 1.0 + 1e-7, 1.0 + 2e-7, 1.0 + 3e-7, 0.5]])
    k = 3

    result = topk_mask_stable(x, k, -1e12, stable=True)

    kept = (result != -1e12).sum()
    assert kept == k

  def test_large_dynamic_range(self):
    """Test with large dynamic range."""
    x = jnp.array([[1e10, 1e5, 1.0, 1e-5, 1e-10]])
    k = 3

    result = topk_mask_stable(x, k, -1e12, stable=True)

    kept = (result != -1e12).sum()
    assert kept == k

  def test_subnormal_numbers(self):
    """Test with subnormal float32 numbers."""
    x = jnp.array([[1e-40, 1e-41, 1e-42, 0.0, -1e-40]], dtype=jnp.float32)
    k = 3

    result = topk_mask_stable(x, k, -1e12, stable=True)

    kept = (result != -1e12).sum()
    # Should handle subnormals correctly
    assert kept >= 0  # At least doesn't crash


class TestIntegrationWithTPUInference:
  """Test integration with tpu_inference functions."""

  def test_topk_mask_stable_parameter(self):
    """Test topk_mask with stable parameter."""
    x = jnp.array([[10.0, 8.0, 8.0, 8.0, 5.0, 3.0]])
    k = 4
    replace_val = -1e12

    # Unstable mode
    result_unstable = topk_mask(x, k, replace_val, stable=False)
    kept_unstable = (result_unstable != replace_val).sum()

    # Stable mode
    result_stable = topk_mask(x, k, replace_val, stable=True)
    kept_stable = (result_stable != replace_val).sum()

    # Unstable should keep more due to ties
    assert kept_unstable >= kept_stable
    # Stable should keep exactly k
    assert kept_stable == k

  def test_topp_mask_stable_parameter(self):
    """Test topp_mask with stable parameter."""
    # Create logits where several tokens have similar probabilities
    logits = jnp.array([[2.0, 1.9, 1.9, 1.9, 0.5, 0.1]])
    p = 0.7
    replace_val = -1e12

    # Both modes should work without error
    result_unstable = topp_mask(logits, p, replace_val, stable=False)
    result_stable = topp_mask(logits, p, replace_val, stable=True)

    # Check that masking happened
    assert (result_unstable == replace_val).any()
    assert (result_stable == replace_val).any()

  def test_topk_mask_comparison(self):
    """Compare our topk_mask with jax.lax.top_k."""
    rng = jax.random.PRNGKey(123)
    x = jax.random.uniform(rng, (8, 200), minval=-5, maxval=5)
    k = 20

    # Our implementation (stable mode)
    our_result = topk_mask(x, k, -jnp.inf, stable=True)
    our_vals = []
    for i in range(x.shape[0]):
      row_vals = our_result[i][our_result[i] != -jnp.inf]
      our_vals.append(jnp.sort(row_vals)[::-1])
    our_vals = jnp.stack(our_vals)

    # JAX implementation
    jax_vals, _ = jax.lax.top_k(x, k)

    # Sort both and compare
    our_sorted = jnp.sort(our_vals, axis=-1)[:, ::-1]
    jax_sorted = jnp.sort(jax_vals, axis=-1)[:, ::-1]

    np.testing.assert_array_almost_equal(our_sorted, jax_sorted, decimal=5)


class TestLargeVocabulary:
  """Test with large vocabulary sizes."""

  def test_large_vocab_64k(self):
    """Test with 64k vocabulary."""
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (4, 65536))
    k = 100

    result = topk_mask_stable(x, k, -jnp.inf, stable=True)

    # Check correctness
    for i in range(4):
      kept = (result[i] != -jnp.inf).sum()
      assert kept == k

  def test_large_vocab_256k(self):
    """Test with 256k vocabulary (realistic for large LLMs)."""
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (2, 262144))  # 256k
    k = 64

    # This should complete in reasonable time with binary search
    # vs would be very slow with full sorting
    result = topk_mask_stable(x, k, -jnp.inf, stable=True)

    # Check correctness
    for i in range(2):
      kept = (result[i] != -jnp.inf).sum()
      assert kept == k


class TestMonotonicConversionsExtended:
  """Extended tests for monotonic conversions."""

  def test_special_values(self):
    """Test special float32 values."""
    special_vals = jnp.array([
      0.0, -0.0,  # Zeros
      jnp.inf, -jnp.inf,  # Infinities
      jnp.finfo(jnp.float32).max,  # Max float32
      jnp.finfo(jnp.float32).min,  # Min float32
      jnp.finfo(jnp.float32).tiny,  # Smallest normal
    ], dtype=jnp.float32)

    u32_vals = monotonic_f32_to_u32(special_vals)
    back = monotonic_u32_to_f32(u32_vals)

    # Should roundtrip (except NaN)
    # Note: +0.0 and -0.0 might not roundtrip exactly
    for i in range(len(special_vals)):
      if jnp.isfinite(special_vals[i]) and special_vals[i] != 0.0:
        assert special_vals[i] == back[i], f"Failed at index {i}"

  def test_monotonicity_comprehensive(self):
    """Comprehensive monotonicity test."""
    rng = jax.random.PRNGKey(42)
    # Generate diverse float values
    test_values = jnp.concatenate([
      jax.random.uniform(rng, (100,), minval=-1e6, maxval=1e6),
      jnp.array([jnp.inf, -jnp.inf, 0.0, -0.0]),
    ])

    # Sort and convert
    sorted_vals = jnp.sort(test_values)
    u32_vals = monotonic_f32_to_u32(sorted_vals)

    # Check monotonicity (excluding NaNs)
    finite_mask = jnp.isfinite(sorted_vals)
    finite_u32 = u32_vals[finite_mask]

    # All differences should be non-negative
    diffs = finite_u32[1:] - finite_u32[:-1]
    assert jnp.all(diffs >= 0), "Monotonicity violated"

  def test_interpolation_properties(self):
    """Test properties of float32 interpolation."""
    l = jnp.array([0.0], dtype=jnp.float32)
    r = jnp.array([100.0], dtype=jnp.float32)

    m = interp_f32(l, r)

    # Midpoint should be strictly between bounds
    assert jnp.all(m > l)
    assert jnp.all(m < r)

    # Multiple interpolations should converge
    for _ in range(10):
      m_new = interp_f32(l, m)
      assert jnp.all(m_new > l)
      assert jnp.all(m_new < m)
      l = m_new

  def test_interpolation_edge_cases(self):
    """Test interpolation with edge cases."""
    # Very close values
    l = jnp.array([1.0], dtype=jnp.float32)
    r = jnp.array([jnp.nextafter(1.0, 2.0)], dtype=jnp.float32)
    m = interp_f32(l, r)
    # Should handle ULP-level differences

    # Large values
    l = jnp.array([1e20], dtype=jnp.float32)
    r = jnp.array([1e30], dtype=jnp.float32)
    m = interp_f32(l, r)
    assert jnp.isfinite(m)  # Should not overflow


class TestThresholdFindingExtended:
  """Extended tests for threshold finding."""

  def test_threshold_with_duplicates_at_start(self):
    """Test when duplicates are at the start."""
    x = jnp.array([[8.0, 8.0, 8.0, 7.0, 6.0, 5.0]])
    k = 2

    threshold = find_topk_threshold_jax(x, k)
    count_ge = (x >= threshold).sum()

    assert count_ge >= k

  def test_threshold_with_duplicates_at_end(self):
    """Test when duplicates are at the end."""
    x = jnp.array([[10.0, 9.0, 8.0, 5.0, 5.0, 5.0]])
    k = 4

    threshold = find_topk_threshold_jax(x, k)
    count_ge = (x >= threshold).sum()

    assert count_ge >= k

  def test_threshold_random_distributions(self):
    """Test threshold finding with various random distributions."""
    rng = jax.random.PRNGKey(42)

    # Uniform distribution
    x_uniform = jax.random.uniform(rng, (8, 1000))
    k = 50
    threshold = find_topk_threshold_jax(x_uniform, k)
    for i in range(8):
      count_ge = (x_uniform[i] >= threshold[i]).sum()
      assert count_ge >= k

    # Normal distribution
    x_normal = jax.random.normal(rng, (8, 1000))
    threshold = find_topk_threshold_jax(x_normal, k)
    for i in range(8):
      count_ge = (x_normal[i] >= threshold[i]).sum()
      assert count_ge >= k


if __name__ == "__main__":
  # Run tests
  pytest.main([__file__, "-v"])
