"""Tallax TAX (TPU Acceleration eXtensions) module.

Public API for TPU-optimized operations.

This module provides JAX.lax-compatible operations optimized for TPU:
- sort: jax.lax.sort counterpart
- top_k: jax.lax.top_k counterpart
- cumsum: jax.lax.cumsum counterpart
- take_along_axis: jax.numpy.take_along_axis counterpart
"""

# Import from _src modules using JAX's pattern
# The "as <name>" syntax is required for proper re-export
from tallax._src.gather import take_along_axis as take_along_axis
from tallax._src.sort import sort as sort
from tallax._src.sort import sort_xla_equivalent as sort_xla_equivalent
from tallax._src.top_k import top_k as top_k
from tallax._src.top_k import top_dynamic_k as top_dynamic_k
from tallax._src.cumsum import cumsum as cumsum

__all__ = [
    "take_along_axis",
    "sort",
    "sort_xla_equivalent",
    "top_k",
    "top_dynamic_k",
    "cumsum",
]
