"""Top-K implementation for TPU.

This module provides top-k selection functionality, with support for both
divide-and-filter and pure bitonic implementations.
"""

import functools
import jax
import jax.numpy as jnp
from jax import jit


@functools.partial(
    jit,
    static_argnames=(
        "k",
        "block_token",
        "num_bins",
        "bins_topm_unroll",
        "bins_topm_schedule",
        "interpret",
        "use_pure_bitonic_implementation",
    ),
)
def top_k(
    logits: jax.Array,
    k: int,
    block_token: int | None = None,
    num_bins: int | None = None,
    bins_topm_unroll: int | None = None,
    bins_topm_schedule: tuple[int, ...] | None = None,
    interpret: bool = False,
    use_pure_bitonic_implementation: bool = False,
):
  """Compute top-k elements with guaranteed convergence.

  This function follows the jax.lax.top_k interface, returning the top k values
  and their indices. By default, uses the divide-and-filter algorithm for efficiency,
  but can optionally use a pure bitonic sort implementation.

  Args:
      logits: Input logits of shape [num_tokens, vocab_size].
      k: Number of top elements to find (uniform across all tokens).
      block_token: Number of tokens processed per program block. If None, uses
          default heuristic based on implementation.
      num_bins: Number of bins for parallel operations (divide-and-filter only).
          Defaults to 128.
      bins_topm_unroll: Loop unroll factor for inner loop (divide-and-filter only).
          Defaults to 32.
      bins_topm_schedule: Optional custom search schedule (divide-and-filter only).
          If None, automatically computed.
      interpret: If True, run in CPU interpret mode (default: False).
      use_pure_bitonic_implementation: If True, uses pure bitonic sort implementation
          instead of divide-and-filter (default: False).

  Returns:
      Tuple of (topk_vals, topk_idxs):
          - topk_vals: Top-k values of shape [num_tokens, k].
          - topk_idxs: Top-k indices of shape [num_tokens, k].
  """
  if use_pure_bitonic_implementation:
    # Use pure bitonic sort implementation
    from tallax._src.sort import bitonic_topk_in_vmem
    from tallax._src.utils import NUM_SUBLANES

    if block_token is None:
      block_token = NUM_SUBLANES

    # bitonic_topk_in_vmem returns (values, indices) with num_keys=1, return_argsort=True hidden
    return bitonic_topk_in_vmem(
        logits,
        k=k,
        num_keys=1,
        return_argsort=True,
        descending=True,
        block_token=block_token,
        interpret=interpret,
    )
  else:
    # Use divide-and-filter implementation
    from tallax._src.divide_and_filter_topk import topk as divide_and_filter_topk
    from tallax._src.utils import NUM_LANES, NUM_SUBLANES

    # Set defaults for divide-and-filter parameters
    if block_token is None:
      block_token = NUM_SUBLANES
    if num_bins is None:
      num_bins = NUM_LANES
    if bins_topm_unroll is None:
      bins_topm_unroll = 32

    return divide_and_filter_topk(
        logits=logits,
        k=k,
        block_token=block_token,
        num_bins=num_bins,
        bins_topm_unroll=bins_topm_unroll,
        bins_topm_schedule=bins_topm_schedule,
        interpret=interpret,
    )
