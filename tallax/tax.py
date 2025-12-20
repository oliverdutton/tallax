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
from tallax.divide_and_filter_topk_convergence_theory import calculate_depth_thresholds

@functools.partial(
    jit,
    static_argnames=(
        "k",
        "reduction_dimension",
        "recall_target",
        "reduction_input_size_override",
        "aggregate_to_topk",
        "use_lax_approx_max_k_algorithm",
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
    use_lax_approx_max_k_algorithm: bool = True,
    block_token: int = NUM_SUBLANES,
    bins_topm_unroll: int = 32,
):
    """TPU-optimized approximate top-k using divide-and-filter algorithm.

    Approximates jax.lax.approx_max_k interface using Tallax's divide-and-filter
    top-k algorithm without guaranteed convergence for maximum speed.

    Two algorithm modes available:
      - use_lax_approx_max_k_algorithm=True: Single-pass using TPU-KNN paper's
        recall formula to calculate num_bins.
      - use_lax_approx_max_k_algorithm=False: Adaptive schedule using convergence
        probability thresholds (faster on average).

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
        use_lax_approx_max_k_algorithm: If True, use TPU-KNN single-pass algorithm.
            If False, use adaptive multi-pass with convergence checks (default: True).
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

    # Determine input size for bin calculation
    if reduction_input_size_override > 0:
        input_size = reduction_input_size_override
    else:
        input_size = operand.shape[-1]

    # Algorithm selection: two different approaches
    if use_lax_approx_max_k_algorithm:
        # TPU-KNN paper's recall formula approach
        # E[Recall] = ((L-1)/L)^(K-1) => L ≈ (k-1) / (1 - recall_target)
        if k == 1:
            num_bins = NUM_LANES
        else:
            num_bins_exact = (k - 1) / (1 - recall_target)
            num_bins = ceil_multiple(math.ceil(num_bins_exact), NUM_LANES)
            num_bins = min(num_bins, ceil_multiple(input_size, NUM_LANES))

        # Single-pass schedule
        bins_topm_schedule = (1,)
    else:
        # Tallax convergence probability approach
        # Auto-select num_bins based on k
        num_bins = 128 if k < 16 else 256
        num_bins = min(num_bins, ceil_multiple(input_size, NUM_LANES))

        # Compute schedule based on recall target
        target_yields = (recall_target,) if recall_target <= 0.95 else (0.9, recall_target)
        thresholds = calculate_depth_thresholds(
            k, num_bins, block_size=1, target_yields=target_yields
        )

        # Add 1 to all except last to enable convergence checks (requires m >= 2)
        # Last depth doesn't need +1 since no convergence check follows it
        bins_topm_schedule = tuple(t + 1 for t in thresholds[:-1]) + (thresholds[-1],)

    # Call top_dynamic_k with computed parameters
    vals, idxs, valid, depths, cutoff = top_dynamic_k(
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
    )

    return vals, idxs

__all__ = [
    "sort",
    "top_k",
    "top_dynamic_k",
    "approx_max_k",
    "cumsum",
]
