"""Tallax TAX (TPU Acceleration eXtensions) module.

Public API for TPU-optimized operations.

This module provides JAX.lax-compatible operations optimized for TPU.
The goal is to provide drop-in replacements for JAX operations where
the TPU-optimized implementation offers significant performance benefits.

Exports:
  sort: TPU-optimized sort (jax.lax.sort counterpart).
  top_k: TPU-optimized top-k with guaranteed convergence (jax.lax.top_k counterpart).
  approx_max_k: TPU-optimized approximate top-k (jax.lax.approx_max_k counterpart).
  cumsum: TPU-optimized cumulative sum (jax.lax.cumsum counterpart).
"""

from tallax.tax.bitonic import (
  bitonic_sort_in_vmem as sort,
  bitonic_topk_in_vmem as bitonic_top_k,
)
from tallax.tax.divide_and_filter_topk import topk as top_k
from tallax.tax.cumsum import cumsum
from tallax.tax.approx_max_k import approx_max_k

__all__ = [
  "sort",
  "top_k",
  "bitonic_top_k",
  "approx_max_k",
  "cumsum",
]
