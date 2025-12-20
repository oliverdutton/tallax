"""Approximate top-k implementation matching jax.lax.approx_max_k interface."""

import math
import functools
import jax.numpy as jnp
from jax import jit

from tallax._src.divide_and_filter_topk import top_dynamic_k
from tallax._src.utils import NUM_LANES, NUM_SUBLANES, ceil_multiple
from tallax.divide_and_filter_topk_convergence_theory import calculate_depth_thresholds


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
    """TPU-optimized approximate top-k using divide-and-filter algorithm taking max of bins.

    Approximates jax.lax.approx_max_k interface using Tallax's divide-and-filter
    top-k algorithm without guaranteed convergence for maximum speed.

    Note: Currently limited to:
      - k <= 128 due to bitonic top-k implementation constraints
      - 2D input only
      - reduction_dimension = -1 only

    Args:
        operand: Input array of shape [batch, vocab] to find top-k elements from.
        k: Number of top elements to retrieve.
        reduction_dimension: Axis along which to find top-k. Must be -1 (default: -1).
        recall_target: Expected recall quality in range (0, 1) (default: 0.95).
        reduction_input_size_override: Override for input size in recall calculation.
            When positive, uses this value instead of operand.shape[-1].
            Useful for SPMD/distributed settings (default: -1).
        aggregate_to_topk: When True, returns sorted top-k results. When False,
            returns approximate results which may be unsorted (default: True).
            Note: Currently ignored, always returns sorted results.
        block_token: Tokens per program block, must divide batch size (default: 8).
        bins_topm_unroll: Loop unroll factor for performance (default: 32).

    Returns:
        Tuple of (values, indices):
            - values: Top-k values of shape [batch, k].
            - indices: Top-k indices of shape [batch, k].

    Raises:
        NotImplementedError: If k > 128, operand is not 2D, or reduction_dimension != -1.
    """
    # Validation
    if k > NUM_LANES:
        raise NotImplementedError(f"k={k} > {NUM_LANES} not yet supported")
    if operand.ndim != 2:
        raise NotImplementedError(f"Only 2D input supported, got {operand.ndim}D")
    if reduction_dimension != -1:
        raise NotImplementedError(f"Only reduction_dimension=-1 supported, got {reduction_dimension}")

    input_size = reduction_input_size_override if reduction_input_size_override > 0 else operand.shape[-1]

    # Uses TPU-KNN paper's recall formula 
    num_bins = math.ceil((k-1)/(1-recall_target)) if k>1 else 1
    bins_topm_schedule = (1,)
    num_bins = ceil_multiple(num_bins, NUM_LANES)
    # incase whole input needs top-k 
    num_bins = min(num_bins, ceil_multiple(input_size, NUM_LANES))
    return top_dynamic_k(
        operand,
        k=k,
        max_k=k,
        block_token=block_token,
        num_bins=num_bins,
        bins_topm_unroll=bins_topm_unroll,
        bins_topm_schedule=bins_topm_schedule,
        guarantee_convergence=False,
        replace_val=None,
        interpret=False,
    )[:2]
