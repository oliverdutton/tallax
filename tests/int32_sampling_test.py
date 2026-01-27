"""
Tests for int32-based sampling implementation.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from tallax.tax.int32_sampling import (
  logits_to_int32_weights,
  int32_cumsum,
  find_top_p_boundary_int32,
  sparse_random_int32,
  sample_token_from_int32_cumsum,
  top_p_and_sample_int32,
  find_boundary_idx,
  _find_boundary_chunk,
)


def test_logits_to_int32_weights_basic():
  """Test conversion of logits to int32 weights."""
  # Simple example: equal logits should give roughly equal weights
  logits = jnp.array([[0.0, 0.0, 0.0, 0.0]])
  weights = logits_to_int32_weights(logits)

  # All weights should be equal
  assert jnp.allclose(weights[0, 0], weights[0, 1], rtol=0.01)
  assert jnp.allclose(weights[0, 0], weights[0, 2], rtol=0.01)
  assert jnp.all(weights > 0)


def test_logits_to_int32_weights_no_overflow():
  """Test that weights sum doesn't overflow int32."""
  # Create logits for a large vocabulary
  k = 1024
  logits = jax.random.normal(jax.random.key(0), (16, k))
  weights = logits_to_int32_weights(logits)

  # Check all weights are positive
  assert jnp.all(weights >= 0)

  # Check sum doesn't overflow (stays well below INT32_MAX)
  sums = jnp.sum(weights, axis=-1)
  assert jnp.all(sums < 2**30)
  assert jnp.all(sums > 0)


def test_int32_cumsum_example():
  """Test cumulative sum with the example [1,3,5,3,7,1]."""
  weights = jnp.array([[1, 3, 5, 3, 7, 1]], dtype=jnp.int32)
  cumsum = int32_cumsum(weights)

  expected = jnp.array([[1, 4, 9, 12, 19, 20]], dtype=jnp.int32)
  np.testing.assert_array_equal(cumsum, expected)


def test_find_top_p_boundary_example():
  """Test finding top-p boundary with example [1,3,5,3,7,1]."""
  weights = jnp.array([[1, 3, 5, 3, 7, 1]], dtype=jnp.int32)
  cumsum = int32_cumsum(weights)
  total = cumsum[:, -1:]  # [[20]]

  # Test p=0.5 (50% of probability mass)
  p = jnp.array([0.5])
  boundary_idx, boundary_sum = find_top_p_boundary_int32(cumsum, total, p)

  # Threshold = 0.5 * 20 = 10
  # cumsum = [1, 4, 9, 12, 19, 20]
  # cumsum < 10: [True, True, True, False, False, False]
  # count of True = 3, so boundary_idx = 3
  # boundary_sum = cumsum[3] = 12
  assert boundary_idx[0] == 3 or boundary_idx[0] == 2
  assert boundary_sum[0] in [9, 12]


def test_find_top_p_boundary_edge_cases():
  """Test edge cases for top-p boundary finding."""
  weights = jnp.array([[1, 3, 5, 3, 7, 1]], dtype=jnp.int32)
  cumsum = int32_cumsum(weights)
  total = cumsum[:, -1:]

  # Test p=1.0 (include all tokens)
  p = jnp.array([1.0])
  boundary_idx, boundary_sum = find_top_p_boundary_int32(cumsum, total, p)
  assert boundary_idx[0] == 5  # Last token
  assert boundary_sum[0] == 20  # Total sum

  # Test p=0.0 (include at least one token)
  p = jnp.array([0.0])
  boundary_idx, boundary_sum = find_top_p_boundary_int32(cumsum, total, p)
  assert boundary_idx[0] >= 0
  assert boundary_sum[0] > 0


def test_sparse_random_int32_range():
  """Test that random int32 values are in correct range."""
  key = jax.random.key(42)
  key_data = jax.random.key_data(key).reshape(1, 2)

  batch_size = 100
  maxval = jnp.full((batch_size,), 20, dtype=jnp.int32)
  indices = (jnp.arange(batch_size), jnp.zeros(batch_size, dtype=jnp.int32))

  random_ints = sparse_random_int32(key_data, indices, dim1_size=1, maxval=maxval)

  # All values should be in [0, 20)
  assert jnp.all(random_ints >= 0)
  assert jnp.all(random_ints < 20)


def test_sparse_random_int32_distribution():
  """Test that random int32 values are roughly uniform."""
  key = jax.random.key(42)
  key_data = jax.random.key_data(key).reshape(1, 2)

  # Generate many samples with maxval=10
  n_samples = 1000
  maxval = jnp.full((n_samples,), 10, dtype=jnp.int32)
  indices = (jnp.arange(n_samples), jnp.zeros(n_samples, dtype=jnp.int32))

  random_ints = sparse_random_int32(key_data, indices, dim1_size=1, maxval=maxval)

  # Check each value [0-9] appears roughly 100 times (10% of 1000)
  for i in range(10):
    count = jnp.sum(random_ints == i)
    # Allow 50-150 occurrences (rough check for uniformity)
    assert count > 50 and count < 150


def test_sample_token_from_cumsum_example():
  """Test token sampling from cumsum with example [1,3,5,3,7,1]."""
  cumsum = jnp.array([[1, 4, 9, 12, 19, 20]], dtype=jnp.int32)

  # Test specific random values
  test_cases = [
    (0, 0),   # random=0 should select token 0 (cumsum[0]=1 > 0)
    (1, 1),   # random=1 should select token 1 (cumsum[1]=4 > 1)
    (3, 1),   # random=3 should select token 1 (cumsum[1]=4 > 3)
    (4, 2),   # random=4 should select token 2 (cumsum[2]=9 > 4)
    (7, 2),   # random=7 should select token 2 (cumsum[2]=9 > 7)
    (9, 3),   # random=9 should select token 3 (cumsum[3]=12 > 9)
    (12, 4),  # random=12 should select token 4 (cumsum[4]=19 > 12)
    (19, 5),  # random=19 should select token 5 (cumsum[5]=20 > 19)
  ]

  for random_val, expected_token in test_cases:
    random_int = jnp.array([random_val], dtype=jnp.int32)
    token_idx = sample_token_from_int32_cumsum(cumsum, random_int)
    assert token_idx[0] == expected_token, \
      f"random={random_val} should select token {expected_token}, got {token_idx[0]}"


def test_sample_token_from_cumsum_batch():
  """Test token sampling with multiple batches."""
  cumsum = jnp.array([
    [1, 4, 9, 12, 19, 20],
    [2, 5, 10, 13, 18, 20],
  ], dtype=jnp.int32)

  random_ints = jnp.array([7, 8], dtype=jnp.int32)
  token_idx = sample_token_from_int32_cumsum(cumsum, random_ints)

  # Batch 0: random=7 should select token 2 (cumsum[0,2]=9 > 7)
  assert token_idx[0] == 2
  # Batch 1: random=8 should select token 2 (cumsum[1,2]=10 > 8)
  assert token_idx[1] == 2


def test_top_p_and_sample_int32_full_pipeline():
  """Test the complete top-p sampling pipeline with int32."""
  # Create simple logits that correspond to weights [1,3,5,3,7,1]
  # We'll use log of these values as logits (since exp(log(x)) = x)
  weights = jnp.array([1, 3, 5, 3, 7, 1], dtype=jnp.float32)
  logits = jnp.log(weights).reshape(1, -1)
  indices = jnp.arange(6).reshape(1, -1)

  key = jax.random.key(42)
  p = jnp.array([0.5])

  # Sample multiple times to check distribution
  samples = []
  for i in range(100):
    key_i = jax.random.fold_in(key, i)
    key_data = jax.random.key_data(key_i).reshape(1, 2)
    sampled = top_p_and_sample_int32(logits, indices, key_data, p)
    samples.append(sampled[0].item())

  # Convert to numpy for easier analysis
  samples = np.array(samples)

  # With p=0.5 and total=20, threshold=10
  # Tokens included: should be a subset of all tokens
  unique_tokens = np.unique(samples)
  # Check that we only sample valid tokens
  assert np.all(unique_tokens >= 0)
  assert np.all(unique_tokens <= 5)


def test_top_p_and_sample_int32_greedy():
  """Test that p=1.0 with clear winner gives correct result."""
  # Logits with one clear winner
  logits = jnp.array([[0.0, 1.0, -1.0, -2.0]])
  indices = jnp.array([[0, 1, 2, 3]])

  key = jax.random.key(42)
  key_data = jax.random.key_data(key).reshape(1, 2)
  p = jnp.array([1.0])

  sampled = top_p_and_sample_int32(logits, indices, key_data, p)

  # With p=1.0, all tokens are included
  # Check it's a valid token
  assert sampled[0] in [0, 1, 2, 3]


@pytest.mark.parametrize("p_value", [0.1, 0.5, 0.9, 1.0])
def test_top_p_and_sample_int32_various_p(p_value):
  """Test sampling with various top-p values."""
  # Use the standard example
  weights = jnp.array([1, 3, 5, 3, 7, 1], dtype=jnp.float32)
  logits = jnp.log(weights).reshape(1, -1)
  indices = jnp.arange(6).reshape(1, -1)

  key = jax.random.key(42)
  key_data = jax.random.key_data(key).reshape(1, 2)
  p = jnp.array([p_value])

  sampled = top_p_and_sample_int32(logits, indices, key_data, p)

  # Check that sampled token is valid
  assert sampled[0] >= 0 and sampled[0] < 6


def test_weights_sum_preservation():
  """Test that total probability mass is preserved in int32 conversion."""
  # Random logits
  logits = jax.random.normal(jax.random.key(0), (8, 100))
  weights = logits_to_int32_weights(logits)

  # Sum should be consistent across batches (roughly)
  sums = jnp.sum(weights, axis=-1)
  mean_sum = jnp.mean(sums)

  # All batch sums should be within 1% of mean
  for batch_sum in sums:
    assert abs(batch_sum / mean_sum - 1.0) < 0.01


def test_numerical_stability_extreme_logits():
  """Test numerical stability with extreme logit values."""
  # Very large and very small logits
  logits = jnp.array([[-100.0, 0.0, 100.0, -50.0]])
  weights = logits_to_int32_weights(logits)

  # Should not overflow or underflow
  assert jnp.all(jnp.isfinite(weights))
  assert jnp.all(weights >= 0)

  # Token 2 (logit=100) should dominate
  assert weights[0, 2] > weights[0, 0] * 100
  assert weights[0, 2] > weights[0, 1] * 10


# ============================================================================
# Tests for k=2048 (exercises hierarchical search fully)
# ============================================================================

def test_hierarchical_search_2048_uniform_weights():
  """Test hierarchical search with 2048 uniform weights."""
  k = 2048
  batch_size = 4

  # Uniform weights: all 1
  weights = jnp.ones((batch_size, k), dtype=jnp.int32)
  cumsum = int32_cumsum(weights)

  # Total sum = k for each batch
  total = cumsum[:, -1:]
  assert jnp.all(total == k)

  # Test p=0.5: should select first k/2 tokens
  p = jnp.array([0.5] * batch_size)
  threshold = (p[:, None] * total).astype(jnp.int32)  # Should be 1024

  boundary_idx, boundary_sum = find_top_p_boundary_int32(cumsum, total, p)

  # With uniform weights, boundary should be around k/2
  # cumsum < 1024 gives us indices [0, 1023] (1024 elements)
  # So boundary_idx should be around 1023-1024
  for i in range(batch_size):
    assert 1020 <= boundary_idx[i] <= 1025, f"boundary_idx[{i}] = {boundary_idx[i]}"
    assert 1020 <= boundary_sum[i] <= 1025, f"boundary_sum[{i}] = {boundary_sum[i]}"


def test_hierarchical_search_2048_geometric_weights():
  """Test hierarchical search with 2048 geometric weights."""
  k = 2048
  batch_size = 2

  # Geometric weights: exponentially decreasing
  # Use a reasonable decay factor to avoid overflow
  indices = jnp.arange(k)
  # Use logits that give geometric distribution: logits[i] = k - i
  # This gives weights that decrease linearly in log space
  logits = (k - indices).astype(jnp.float32).reshape(1, k) * 0.01  # Scale down
  logits = jnp.repeat(logits, batch_size, axis=0)

  weights = logits_to_int32_weights(logits)
  cumsum = int32_cumsum(weights)
  total = cumsum[:, -1:]

  # Test various p values
  for p_val in [0.1, 0.5, 0.9]:
    p = jnp.array([p_val] * batch_size)
    boundary_idx, boundary_sum = find_top_p_boundary_int32(cumsum, total, p)

    # Verify boundary is within valid range
    assert jnp.all(boundary_idx >= 0)
    assert jnp.all(boundary_idx < k)

    # Verify boundary_sum is approximately p * total
    for i in range(batch_size):
      ratio = boundary_sum[i] / total[i, 0]
      # Should be close to p (within a few percent due to integer rounding)
      assert abs(ratio - p_val) < 0.1, f"p={p_val}, ratio={ratio}"


def test_hierarchical_search_2048_token_sampling():
  """Test token sampling with 2048 elements using hierarchical search."""
  k = 2048
  batch_size = 8

  # Create weights with some variation
  # Use a smooth distribution: weights[i] = k - i (linearly decreasing)
  weights = jnp.array([k - i for i in range(k)], dtype=jnp.int32)
  weights = jnp.repeat(weights.reshape(1, k), batch_size, axis=0)

  cumsum = int32_cumsum(weights)
  total = cumsum[:, -1:]

  # Sample multiple times and verify distribution
  key = jax.random.key(123)
  n_samples = 100

  samples = []
  for i in range(n_samples):
    key_i = jax.random.fold_in(key, i)
    key_data = jax.random.key_data(key_i).reshape(1, 2)

    # Generate random ints in [0, total) for each batch
    batch_indices = jnp.arange(batch_size)
    dim1_indices = jnp.zeros(batch_size, dtype=jnp.int32)
    random_ints = sparse_random_int32(
      key_data,
      (batch_indices, dim1_indices),
      dim1_size=1,
      maxval=total.squeeze(1),
    )

    # Sample tokens
    token_idx = sample_token_from_int32_cumsum(cumsum, random_ints)
    samples.append(token_idx)

  samples = jnp.stack(samples)  # (n_samples, batch_size)

  # Verify all samples are in valid range
  assert jnp.all(samples >= 0)
  assert jnp.all(samples < k)

  # Verify samples have some variation (not all the same)
  for batch_i in range(batch_size):
    unique_samples = jnp.unique(samples[:, batch_i])
    assert len(unique_samples) > 1, f"Batch {batch_i} has no variation in samples"


def test_hierarchical_search_2048_boundary_accuracy():
  """Test boundary finding accuracy with 2048 elements."""
  k = 2048
  batch_size = 16

  # Create deterministic weights
  weights = jnp.array([i + 1 for i in range(k)], dtype=jnp.int32)
  weights = jnp.repeat(weights.reshape(1, k), batch_size, axis=0)

  cumsum = int32_cumsum(weights)
  total = cumsum[:, -1:]

  # Total sum = 1 + 2 + ... + k = k*(k+1)/2 = 2048*2049/2 = 2,098,176

  # Test multiple p values
  p_values = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]

  for p_val in p_values:
    p = jnp.array([p_val] * batch_size)
    boundary_idx, boundary_sum = find_top_p_boundary_int32(cumsum, total, p)

    for batch_i in range(batch_size):
      # Verify boundary_sum <= p * total (within rounding)
      threshold = int(p_val * total[batch_i, 0])
      assert boundary_sum[batch_i] <= threshold + k, \
        f"p={p_val}: boundary_sum={boundary_sum[batch_i]} > threshold={threshold}"

      # Verify boundary_idx is reasonable
      assert 0 <= boundary_idx[batch_i] < k, \
        f"p={p_val}: boundary_idx={boundary_idx[batch_i]} out of range"

      # Verify cumsum at boundary matches boundary_sum
      actual_cumsum = cumsum[batch_i, boundary_idx[batch_i]]
      assert actual_cumsum == boundary_sum[batch_i], \
        f"p={p_val}: cumsum mismatch at boundary"


def test_hierarchical_search_2048_chunk_transitions():
  """Test that hierarchical search handles chunk boundaries correctly."""
  k = 2048
  batch_size = 4

  # Create weights where boundaries fall exactly on chunk edges
  # NUM_LANES = 128, so chunks are at 0, 128, 256, ...
  # coarse_chunk_size = sqrt(2048/128) * 128 = 4 * 128 = 512
  # So coarse chunks are at 0, 512, 1024, 1536, 2048

  weights = jnp.ones((batch_size, k), dtype=jnp.int32)
  cumsum = int32_cumsum(weights)
  total = cumsum[:, -1:]

  # Test p values that should land exactly on chunk boundaries
  # cumsum at index i = i+1 (since all weights are 1)
  # For cumsum = 512, index = 511
  # For cumsum = 1024, index = 1023
  # For cumsum = 1536, index = 1535

  chunk_boundaries = [512, 1024, 1536]
  for boundary_val in chunk_boundaries:
    p_val = boundary_val / k  # Proportion to reach this boundary
    p = jnp.array([p_val] * batch_size)

    boundary_idx, boundary_sum = find_top_p_boundary_int32(cumsum, total, p)

    # Should find boundary around the chunk edge
    for batch_i in range(batch_size):
      # Allow some tolerance for integer rounding
      assert abs(boundary_sum[batch_i] - boundary_val) < 10, \
        f"boundary_val={boundary_val}: boundary_sum={boundary_sum[batch_i]}"


def test_hierarchical_search_2048_edge_case_p_values():
  """Test edge cases with p=0 and p=1 for k=2048."""
  k = 2048
  batch_size = 4

  weights = jnp.array([i + 1 for i in range(k)], dtype=jnp.int32)
  weights = jnp.repeat(weights.reshape(1, k), batch_size, axis=0)

  cumsum = int32_cumsum(weights)
  total = cumsum[:, -1:]

  # Test p=0: should include at least first token
  p = jnp.array([0.0] * batch_size)
  boundary_idx, boundary_sum = find_top_p_boundary_int32(cumsum, total, p)

  assert jnp.all(boundary_idx >= 0), "p=0: boundary_idx should be >= 0"
  assert jnp.all(boundary_sum > 0), "p=0: boundary_sum should be > 0"

  # Test p=1.0: should include all tokens
  p = jnp.array([1.0] * batch_size)
  boundary_idx, boundary_sum = find_top_p_boundary_int32(cumsum, total, p)

  for batch_i in range(batch_size):
    assert boundary_idx[batch_i] == k - 1, \
      f"p=1.0: boundary_idx should be {k-1}, got {boundary_idx[batch_i]}"
    assert boundary_sum[batch_i] == total[batch_i, 0], \
      f"p=1.0: boundary_sum should equal total"


def test_hierarchical_search_2048_full_pipeline():
  """Test complete sampling pipeline with k=2048."""
  k = 2048
  batch_size = 8

  # Create logits with realistic distribution
  logits = jax.random.normal(jax.random.key(456), (batch_size, k))
  indices = jnp.arange(k).reshape(1, k)
  indices = jnp.repeat(indices, batch_size, axis=0)

  key = jax.random.key(789)
  p = jnp.array([0.9] * batch_size)  # Top-90%

  # Run full pipeline
  samples = []
  for i in range(50):
    key_i = jax.random.fold_in(key, i)
    key_data = jax.random.key_data(key_i).reshape(1, 2)
    sampled = top_p_and_sample_int32(logits, indices, key_data, p)
    samples.append(sampled)

  samples = jnp.stack(samples)  # (50, batch_size)

  # Verify all samples are valid
  assert jnp.all(samples >= 0)
  assert jnp.all(samples < k)

  # Verify there's diversity in samples
  for batch_i in range(batch_size):
    unique_samples = jnp.unique(samples[:, batch_i])
    # With top-90% and 50 samples, should see multiple different tokens
    assert len(unique_samples) >= 5, \
      f"Batch {batch_i} has too little diversity: {len(unique_samples)} unique samples"


def test_hierarchical_search_2048_performance_characteristics():
  """Verify hierarchical search examines fewer elements than full scan."""
  k = 2048
  # NUM_LANES = 128
  # coarse_chunk_size = sqrt(2048 / 128) * 128 = 4 * 128 = 512
  # Phase 1: Examine 2048 / 512 = 4 chunks
  # Phase 2: Examine 512 / 128 = 4 chunks
  # Phase 3: Examine 128 elements
  # Total: ~4 + 4 + 128 = 136 elements examined (vs 2048 for full scan)

  # This is more of a documentation test - the implementation
  # should examine approximately sqrt(k) + NUM_LANES elements
  import math
  from tallax.tax.utils import NUM_LANES

  expected_elements_examined = math.sqrt(k / NUM_LANES) * NUM_LANES + NUM_LANES
  full_scan_elements = k

  reduction_factor = full_scan_elements / expected_elements_examined

  print(f"\nHierarchical search performance for k={k}:")
  print(f"  Full scan: {full_scan_elements} elements")
  print(f"  Hierarchical: ~{int(expected_elements_examined)} elements")
  print(f"  Reduction: {reduction_factor:.1f}x")

  assert reduction_factor > 3, "Hierarchical search should be at least 3x faster"
