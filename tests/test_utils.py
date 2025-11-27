"""Utilities for testing tallax operations."""

import jax
import jax.numpy as jnp


@jax.vmap
def check_topk_out(x, outs):
    """Validate top-k outputs for correctness.

    Args:
        x: Input array (1D per vmap)
        outs: Tuple of (values, indices) from top-k

    Returns:
        Boolean indicating if the top-k output is valid
    """
    assert x.ndim == 1
    out_vals, out_indexs = outs
    x_sorted = jnp.sort(x, descending=True)

    k = len(out_vals)
    n = len(x)
    valid = True

    # actual values must match
    valid &= (out_vals == x_sorted[:k]).all()

    # indices map to values correctly
    valid &= (x[out_indexs] == out_vals).all()

    # indices are all in bounds and unique
    i = jnp.unique(out_indexs, size=k, fill_value=-1)
    valid &= ((i >= 0) & (i < n)).all()
    return valid
