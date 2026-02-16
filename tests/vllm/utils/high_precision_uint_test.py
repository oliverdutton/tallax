"""Tests for tallax.vllm.utils.high_precision_uint."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from tallax.constants import SCALE_BITS
from tallax.vllm.utils.high_precision_uint import U48, modulo_u128_u64


@pytest.mark.parametrize("seed", [42, 123, 456])
def test_u48(seed):
  """U48.map_reduce_sum matches i64 sum and < operator is consistent."""
  key = jax.random.PRNGKey(seed)
  scale = 2**SCALE_BITS - 1
  vals = jax.random.randint(key, (4, 512), 0, scale, dtype=jnp.int32)
  u48_sum = U48.map_reduce_sum(vals, max_val=scale)

  with jax.enable_x64(True):
    expected = vals.astype(jnp.int64).sum(axis=1, keepdims=True).astype(jnp.float64)
  np.testing.assert_allclose(u48_sum.to_f32().astype(float), np.array(expected).astype(float), rtol=1e-6)

  # Comparison
  key = jax.random.split(key)[0]
  a_vals = jax.random.randint(key, (10,), 0, scale, dtype=jnp.int32)
  key = jax.random.split(key)[0]
  b_vals = jax.random.randint(key, (10,), 0, scale, dtype=jnp.int32)
  np.testing.assert_array_equal(U48(a_vals, max_val=scale) < U48(b_vals, max_val=scale), a_vals < b_vals)


@pytest.mark.parametrize("seed", [42, 123, 456])
def test_modulo_u128_u64(seed):
  """modulo_u128_u64 matches Python arbitrary precision."""
  key = jax.random.PRNGKey(seed)
  dividend = tuple(jax.random.bits(key, (4, 2, 1), jnp.uint32))
  key = jax.random.split(key)[0]
  divisor_low = jax.random.randint(key, (2, 1), 1, 2**31, dtype=jnp.int32).astype(jnp.uint32)
  divisor = [jnp.zeros_like(divisor_low), divisor_low]

  result_h, result_l = modulo_u128_u64(dividend, divisor)

  d = [np.array(x, dtype=object) for x in dividend]
  val_128 = (d[0] << 96) | (d[1] << 64) | (d[2] << 32) | d[3]
  m = np.array(divisor_low, dtype=object)
  expected = (val_128 % m).astype(np.uint64)
  actual = (np.array(result_h, dtype=np.uint64) << 32) + np.array(result_l, dtype=np.uint64)
  np.testing.assert_array_equal(actual, expected)
