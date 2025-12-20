"""Tallax TAX (TPU Acceleration eXtensions) module.

Public API for TPU-optimized operations.

This module provides JAX.lax-compatible operations optimized for TPU.
The goal is to provide drop-in replacements for JAX operations where
the TPU-optimized implementation offers significant performance benefits.

Exports:
  sort: TPU-optimized sort (jax.lax.sort counterpart).
  top_k: TPU-optimized top-k with guaranteed convergence (jax.lax.top_k counterpart).
  top_dynamic_k: Top-k supporting per-token (dynamic) k values.
  approx_max_k: TPU-optimized approximate top-k (jax.lax.approx_max_k counterpart).
  cumsum: TPU-optimized cumulative sum (jax.lax.cumsum counterpart).
"""
import math
import functools
import jax.numpy as jnp
from jax import jit

from tallax._src.sort import sort
from tallax._src.divide_and_filter_topk import topk as top_k
from tallax._src.divide_and_filter_topk import top_dynamic_k
from tallax._src.cumsum import cumsum
from tallax._src.utils import NUM_LANES, NUM_SUBLANES, ceil_multiple

@functools.partial(
    jit,
    static_argnames=(
        "k",
        "reduction_dimension",
        "recall_target",
        "reduction_input_size_override",
        "aggregate_to_topk",
        "block_token",
        "bins_topm_unroll",
    ),
)
def approx_max_k(
    operand,
    k: int,
    reduction_dimension: int = -1,
    recall_target: float = 0.95,
    reduction_input_size_override: int = -1,
    aggregate_to_topk: bool = True,
    block_token: int = NUM_SUBLANES,
    bins_topm_unroll: int = 32,
):
    """TPU-optimized approximate top-k using divide-and-filter algorithm.

    Approximates jax.lax.approx_max_k interface using Tallax's divide-and-filter
    top-k algorithm. Uses a single bin-pass (bins_topm_schedule=(1,)) without
    guaranteed convergence for maximum speed.

    Note: Currently limited to k <= 128 due to bitonic top-k implementation constraints.

    Args:
        operand: Input array to find top-k elements from.
        k: Number of top elements to retrieve.
        reduction_dimension: Axis along which to find top-k (default: -1).
        recall_target: Expected recall quality in range (0, 1) (default: 0.95).
        reduction_input_size_override: Override for input size in recall calculation.
            When positive, uses this value instead of operand.shape[reduction_dimension].
            Useful for SPMD/distributed settings (default: -1).
        aggregate_to_topk: When True, returns sorted top-k results. When False,
            returns approximate results which may be unsorted (default: True).
            Note: Currently ignored, always returns sorted results.
        block_token: Tokens per program block, must divide batch size (default: 8).
        bins_topm_unroll: Loop unroll factor for performance (default: 32).

    Returns:
        Tuple of (values, indices):
            - values: Top-k values of shape operand.shape with reduction_dimension
                replaced by k.
            - indices: Top-k indices of shape matching values.

    Raises:
        NotImplementedError: If k > 128.
    """
    if k > NUM_LANES:
        raise NotImplementedError(f"k={k} > {NUM_LANES} not yet supported")

    # Calculate number of bins using recall formula from TPU-KNN paper
    # E[Recall] = ((L-1)/L)^(K-1)
    # Solving for L: L = 1 / (1 - recall_target^(1/(k-1)))
    # Approximation: L ≈ (k-1) / (1 - recall_target)
    if reduction_input_size_override > 0:
        input_size = reduction_input_size_override
    else:
        input_size = operand.shape[reduction_dimension]

    if k == 1:
        # Special case: k=1 means 100% recall with any L >= 1
        num_bins = NUM_LANES
    else:
        # Use approximation formula: L ≈ (k-1) / (1 - recall_target)
        num_bins_exact = (k - 1) / (1 - recall_target)
        # Round up to nearest multiple of NUM_LANES for tile alignment
        num_bins = ceil_multiple(math.ceil(num_bins_exact), NUM_LANES)
        # Ensure num_bins doesn't exceed input size
        num_bins = min(num_bins, ceil_multiple(input_size, NUM_LANES))

    # Normalize reduction dimension
    ndim = operand.ndim
    if reduction_dimension < 0:
        reduction_dimension = ndim + reduction_dimension

    # Move reduction dimension to last position if needed
    needs_transpose = reduction_dimension != ndim - 1
    if needs_transpose:
        perm = list(range(ndim))
        perm[reduction_dimension], perm[-1] = perm[-1], perm[reduction_dimension]
        operand = jnp.transpose(operand, perm)

    # Call top_dynamic_k with single-pass schedule and no convergence guarantee
    vals, idxs, valid, depths, cutoff = top_dynamic_k(
        operand,
        k=k,
        max_k=k,
        block_token=block_token,
        num_bins=num_bins,
        bins_topm_unroll=bins_topm_unroll,
        bins_topm_schedule=(1,),  # Single pass as specified
        guarantee_convergence=False,  # Approximate, no guarantee
        replace_val=None,
        interpret=False,
    )

    # Restore original dimension order if transposed
    if needs_transpose:
        vals = jnp.transpose(vals, perm)
        idxs = jnp.transpose(idxs, perm)

    return vals, idxs

__all__ = [
    "sort",
    "top_k",
    "top_dynamic_k",
    "approx_max_k",
    "cumsum",
]
