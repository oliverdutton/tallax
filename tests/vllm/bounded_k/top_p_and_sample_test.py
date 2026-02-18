"""Tests for tallax.vllm.bounded_k.top_p_and_sample.

Tests top_p_mask against the standalone tpu_inference reference implementation.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from tallax.constants import REPLACE_VAL
from tallax.vllm.bounded_k.top_p_and_sample import top_p_mask as pallas_top_p_mask
from tallax.vllm.tpu_inference_sampling_as_standalone_file import (
  topp_mask as tpu_inference_top_p_mask,
)


@pytest.mark.parametrize("shape", [(8, 128), (16, 256), (137, 193)])
@pytest.mark.parametrize("seed", [42, 123])
@pytest.mark.parametrize("p_threshold", [0.1, 0.9, 1.0, None])
def test_top_p_mask(shape, seed, p_threshold):
  """pallas top_p_mask matches tpu_inference_top_p_mask exactly."""
  key = jax.random.key(seed)
  key, logits_key, p_key = jax.random.split(key, 3)

  logits = jax.random.normal(logits_key, shape, dtype=jnp.float32)

  if p_threshold is None:
    p_array = jax.random.uniform(p_key, shape[:1], dtype=jnp.float32)
  else:
    p_array = jnp.full(shape[:1], p_threshold, dtype=jnp.float32)

  sort_indices = jnp.argsort(logits, axis=1, descending=True)
  sorted_logits = jnp.take_along_axis(logits, sort_indices, axis=1)

  result_sorted = pallas_top_p_mask(
    topk_logits=sorted_logits.T, p=p_array, axis=0,
  ).T

  inverse_sort_indices = jnp.argsort(sort_indices, axis=1)
  result_original_order = jnp.take_along_axis(result_sorted, inverse_sort_indices, axis=1)

  result_ref = jnp.zeros_like(logits)
  for i in range(shape[0]):
    result_ref = result_ref.at[i].set(
      tpu_inference_top_p_mask(logits[i:i+1], float(p_array[i]), REPLACE_VAL)[0]
    )

  np.testing.assert_array_equal(result_original_order, result_ref)
