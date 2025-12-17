"""Standalone take_along_axis_arrays implementation using only JAX and functools."""

import functools
import jax
import jax.numpy as jnp
from jax import lax


# TPU hardware constants
NUM_SUBLANES = 8
NUM_LANES = 128


def pad(arr, tile_shape, val=0):
    """Pad array to be multiple of tile_shape dimensions.

    Args:
        arr: Input array to pad.
        tile_shape: Target tile shape (tuple of ints).
        val: Padding value (default: 0).

    Returns:
        Padded array.
    """
    if len(tile_shape) != arr.ndim:
        raise ValueError(
            f"tile_shape length {len(tile_shape)} must match array ndim {arr.ndim}"
        )

    # Calculate padding for each dimension
    pad_widths = []
    for dim_size, block_size in zip(arr.shape, tile_shape):
        # Calculate target size as next multiple of block_size
        target_size = ((dim_size + block_size - 1) // block_size) * block_size
        pad_size = target_size - dim_size
        pad_widths.append((0, pad_size))

    # Return early if no padding needed
    if all(w == (0, 0) for w in pad_widths):
        return arr

    return jnp.pad(arr, pad_widths, mode='constant', constant_values=val)


def take_along_axis_arrays(val, idx, axis):
    """Gather values from val array using indices in idx array.

    This function performs a batched gather operation that's optimized for TPU
    hardware. It processes data in tiles of size (NUM_SUBLANES, NUM_LANES).

    Args:
        val: Values array to gather from.
        idx: Indices array specifying which values to gather.
        axis: Axis along which to gather.

    Returns:
        Gathered values with shape matching idx.shape.
    """
    shape = idx.shape
    tile_shape = (NUM_SUBLANES, NUM_LANES)
    val, idx = (pad(x, tile_shape, val=0) for x in (val, idx))

    def _gather_arrays(val, idx):
        # Initialize accumulators
        accumulators = [
            jnp.zeros(tile_shape, dtype=val.dtype)
            for _ in range(idx.shape[axis] // tile_shape[axis])
        ]
        for val_offset in range(0, val.shape[axis], tile_shape[axis]):
            # Load values for this block once
            val_tile = lax.slice_in_dim(val, val_offset, val_offset+tile_shape[axis], axis=axis)

            # Apply to all K blocks
            for idx_offset in range(0, idx.shape[axis], tile_shape[axis]):
                idx_tile = lax.slice_in_dim(idx, idx_offset, idx_offset+tile_shape[axis], axis=axis)
                mask = (idx_tile >= val_offset) & (idx_tile < val_offset + tile_shape[axis])
                gather_tile = jnp.take_along_axis(
                    val_tile,
                    (idx_tile - val_offset) % tile_shape[axis],
                    axis=axis
                )
                i = idx_offset // tile_shape[axis]
                accumulators[i] = jnp.where(mask, gather_tile, accumulators[i])
        return jnp.concatenate(accumulators, axis=axis)

    batch_axis = 1 - axis
    assert val.shape[batch_axis]==idx.shape[batch_axis]
    return jnp.concatenate(
        [_gather_arrays(v, i)
         for v, i in zip(*map(lambda arr: jnp.split(
             arr, arr.shape[batch_axis] // tile_shape[batch_axis], axis=batch_axis), (val, idx)))
        ],
        axis=batch_axis
    )[:shape[0], :shape[1]]
