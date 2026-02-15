"""E2E test for tallax.vllm.arbitrary_k.kernel against the reference implementation.

Uses debug intermediates from the reference to verify each stage of the kernel.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from tallax.vllm.reference import reference_topk_topp_mask_and_sample
from tallax.vllm.arbitrary_k.kernel import topk_topp_mask_and_sample as arbitrary_k_sample
from tallax.tax.utils import is_cpu_platform


def _make_inputs(seed, batch_size, vocab_size):
  key = jax.random.PRNGKey(seed)
  keys = jax.random.split(key, 6)
  logits = jax.random.normal(keys[0], (batch_size, vocab_size), dtype=jnp.float32)
  k = jax.random.randint(keys[1], (batch_size,), 1, min(64, vocab_size), dtype=jnp.int32)
  p = jax.random.uniform(keys[2], (batch_size,), dtype=jnp.float32, minval=0.1, maxval=1.0)
  temperature = 10 ** jax.random.normal(keys[3], (batch_size,), dtype=jnp.float32)
  temperature = jnp.clip(temperature, 0.1, 10.0)
  rng_key = keys[4]
  return logits, k, p, temperature, rng_key


@pytest.mark.parametrize("seed", [42, 123, 456])
@pytest.mark.skipif(is_cpu_platform(), reason="arbitrary_k kernel requires TPU/GPU")
def test_arbitrary_k_vs_reference(seed):
  """arbitrary_k kernel matches reference output and debug intermediates."""
  logits, k, p, temperature, rng_key = _make_inputs(seed, 4, 256)

  ref_result, ref_debug = reference_topk_topp_mask_and_sample(
    logits, rng_key, k, p, temperature, debug=True
  )
  kernel_result, kernel_debug = arbitrary_k_sample(
    logits, rng_key, k, p, temperature, stable=True, debug=True
  )

  # Final tokens match
  np.testing.assert_array_equal(kernel_result, ref_result)

  # Greedy argmax matches
  np.testing.assert_array_equal(
    kernel_debug["greedy_sampled"][0, 0],
    ref_debug["greedy_sampled"][0],
  )

  # Next tokens match
  np.testing.assert_array_equal(
    kernel_debug["next_tokens"][0, 0],
    ref_debug["next_tokens"][0],
  )

  # Top-p nonzero count matches
  np.testing.assert_array_equal(
    kernel_debug["topp_nonzero_count"][0, 0],
    ref_debug["topp_nonzero_count"][0],
  )
