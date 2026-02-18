"""Tests for tallax.vllm.arbitrary_k.topk_mask."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from tallax.constants import REPLACE_VAL
from tallax.vllm.arbitrary_k.topk_mask import topk_mask_pallas
from tallax.tax.utils import is_cpu_platform


@pytest.mark.parametrize("seed", [42, 123, 456])
@pytest.mark.parametrize("k_val", [1, 10, 50])
@pytest.mark.skipif(is_cpu_platform(), reason="topk_mask_pallas requires TPU/GPU")
def test_topk_mask(seed, k_val):
  """topk_mask keeps exactly k correct top values when stable=True."""
  key = jax.random.PRNGKey(seed)
  batch_size, vocab_size = 4, 256
  logits = jax.random.normal(key, (batch_size, vocab_size), dtype=jnp.float32)
  k = jnp.full((batch_size,), k_val, dtype=jnp.int32)
  masked = topk_mask_pallas(logits, k, stable=True)

  # Check count
  counts = (masked != REPLACE_VAL).sum(axis=1)
  np.testing.assert_array_equal(counts, jnp.full((batch_size,), k_val))

  # Check values match jax.lax.top_k
  for b in range(batch_size):
    ref_vals, _ = jax.lax.top_k(logits[b], k_val)
    actual_vals = jnp.sort(masked[b][masked[b] != REPLACE_VAL])[::-1]
    np.testing.assert_allclose(actual_vals[:k_val], ref_vals, atol=1e-5)
