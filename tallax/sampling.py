"""Tallax sampling module.

Public API for TPU-optimized sampling operations.
"""

# Import from _src module using JAX's pattern
# The "as <name>" syntax is required for proper re-export
from tallax._src.sampling import topp_mask as topp_mask
from tallax._src.sampling import top_p_and_sample_jax_inner as top_p_and_sample_jax_inner
from tallax._src.sampling import top_p_and_sample_kernel as top_p_and_sample_kernel
from tallax._src.sampling import top_p_and_sample as top_p_and_sample

__all__ = [
    "topp_mask",
    "top_p_and_sample_jax_inner",
    "top_p_and_sample_kernel",
    "top_p_and_sample",
]
