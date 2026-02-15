"""Tests for tallax.vllm.utils.binary_search."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from tallax.vllm.utils.binary_search import (
  binary_search,
  monotonic_f32_to_u32,
  monotonic_u32_to_f32,
)


@pytest.mark.parametrize("seed", [42, 123, 456])
def test_monotonic_f32_u32(seed):
  """monotonic_f32_to_u32 roundtrips exactly and preserves ordering."""
  key = jax.random.PRNGKey(seed)
  vals = jax.random.normal(key, (100,), dtype=jnp.float32)
  roundtripped = monotonic_u32_to_f32(monotonic_f32_to_u32(vals))
  np.testing.assert_array_equal(vals, roundtripped)

  sorted_vals = jnp.sort(vals)
  sorted_u32 = monotonic_f32_to_u32(sorted_vals)
  assert jnp.all(sorted_u32[1:] >= sorted_u32[:-1])


@pytest.mark.parametrize("target", [0.0, 0.5, -1.5, 3.14])
def test_binary_search(target):
  """Binary search converges to the correct threshold."""
  target_arr = jnp.array([[target]], dtype=jnp.float32)
  lo = jnp.full((1, 1), -100.0, jnp.float32)
  hi = jnp.full((1, 1), 100.0, jnp.float32)
  _, threshold, _ = binary_search(lambda pivot: pivot < target_arr, lo, hi, num_iter=32)
  np.testing.assert_allclose(float(threshold), target, atol=1e-6)
