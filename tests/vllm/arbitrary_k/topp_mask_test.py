"""Tests for tallax.vllm.arbitrary_k.topp_mask."""

import pytest
import jax
import jax.numpy as jnp

from tallax.vllm.arbitrary_k.topp_mask import topp_mask


@pytest.mark.parametrize("seed", [42, 123, 456])
@pytest.mark.parametrize("p_val", [0.1, 0.5, 0.9, 1.0])
def test_topp_mask(seed, p_val):
  """topp_mask returns at least one nonzero token per batch."""
  key = jax.random.PRNGKey(seed)
  batch_size, vocab_size = 4, 256
  logits = jax.random.normal(key, (batch_size, vocab_size), dtype=jnp.float32)
  p = jnp.full((batch_size, 1), p_val, dtype=jnp.float32)
  result = topp_mask(logits, p)
  nonzero_count = (result != 0).sum(axis=1)
  assert jnp.all(nonzero_count > 0), f"topp_mask should keep at least 1 token, got {nonzero_count}"
