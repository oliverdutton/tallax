"""Tallax sampling module.

Public API for TPU-optimized sampling operations.
"""

from tallax._src.sampling import topk_topp_and_sample
from tallax._src.sampling import top_p_and_sample

__all__ = [
    "topk_topp_and_sample",
    "top_p_and_sample",
]
