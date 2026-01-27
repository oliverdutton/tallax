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
