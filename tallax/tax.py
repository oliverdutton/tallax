"""Tallax TAX (TPU Acceleration eXtensions) module.

Public API for TPU-optimized operations.
"""

# Import from _src modules using JAX's pattern
# The "as <name>" syntax is required for proper re-export
from tallax._src.gather import take_along_axis as take_along_axis
from tallax._src.sort import sort as sort
from tallax._src.sort import sort_xla_equivalent as sort_xla_equivalent
from tallax._src.top_k import top_k as top_k
from tallax._src.top_k import top_dynamic_k as top_dynamic_k
from tallax._src.cumsum import cumsum as cumsum
from tallax._src.bitonic_topk import bitonic_topk as bitonic_topk
from tallax._src.bitonic_topk import pallas_compatible_bitonic_topk as pallas_compatible_bitonic_topk
from tallax._src.bitonic_topk import top1 as top1

__all__ = [
    "take_along_axis",
    "sort",
    "top_k",
    "top_dynamic_k",
    "sort_xla_equivalent",
    "cumsum",
    "bitonic_topk",
    "pallas_compatible_bitonic_topk",
    "top1",
]
