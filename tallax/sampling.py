"""Tallax sampling module.

Public API for TPU-optimized sampling operations.
"""

from tallax._src.sampling import sample
from tallax._src.sampling import top_p_and_sample

__all__ = [
    "sample",
    "top_p_and_sample",
]
