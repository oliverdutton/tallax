"""Top-K implementation for TPU.

This module provides top-k selection functionality following jax.lax.top_k interface,
with support for both divide-and-filter and pure bitonic implementations.
"""

import functools
import jax
import jax.numpy as jnp
from jax import jit


@functools.partial(
    jit,
    static_argnames=("k",),
)
def top_k(
    operand: jax.Array,
    k: int,
):
  """Compute top-k elements with guaranteed convergence.

  This function follows the jax.lax.top_k interface, returning the top k values
  and their indices in descending order. Uses the divide-and-filter algorithm
  for efficiency.

  Args:
      operand: Input array of shape [num_tokens, vocab_size].
      k: Number of top elements to find (uniform across all tokens).

  Returns:
      Tuple of (topk_vals, topk_idxs):
          - topk_vals: Top-k values of shape [num_tokens, k] in descending order.
          - topk_idxs: Top-k indices of shape [num_tokens, k].
  """
  # Use divide-and-filter implementation (default)
  from tallax.divide_and_filter_topk.topk import topk as divide_and_filter_topk
  from tallax.tax.utils import NUM_LANES, NUM_SUBLANES, is_cpu_platform

  # Set defaults
  block_token = NUM_SUBLANES
  num_bins = NUM_LANES
  bins_topm_unroll = 32
  interpret = is_cpu_platform()

  return divide_and_filter_topk(
      logits=operand,
      k=k,
      block_token=block_token,
      num_bins=num_bins,
      bins_topm_unroll=bins_topm_unroll,
      bins_topm_schedule=None,
      interpret=interpret,
  )
