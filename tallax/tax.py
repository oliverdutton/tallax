"""Tallax TAX (TPU Acceleration eXtensions) module.

Public API for TPU-optimized operations.

This module provides JAX.lax-compatible operations optimized for TPU:
- sort: jax.lax.sort counterpart
- top_k: jax.lax.top_k counterpart
- cumsum: jax.lax.cumsum counterpart
"""

# Import from _src modules using JAX's pattern
# The "as <name>" syntax is required for proper re-export
from tallax._src.sort import sort as sort
from tallax._src.top_k import top_k as top_k
from tallax._src.top_k import top_dynamic_k as top_dynamic_k
from tallax._src.cumsum import cumsum as cumsum

__all__ = [
    "sort",
    "top_k",
    "top_dynamic_k",
    "cumsum",
]
