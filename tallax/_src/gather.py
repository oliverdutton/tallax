
import functools
import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax._src.utils import NUM_LANES, NUM_SUBLANES, pad

def take_along_axis_arrays(val, idx, axis):
  shape = idx.shape
  tile_shape = (NUM_SUBLANES, NUM_LANES)
  val, idx = (pad(x, tile_shape, val=0) for x in (val, idx))
  def _gather_arrays(val, idx):
    # Initialize accumulators
    num_idx_tiles = idx.shape[axis] // tile_shape[axis]
    accumulators = [
        jnp.zeros(tile_shape, dtype=val.dtype)
        for _ in range(num_idx_tiles)
    ]

    # Split arrays into tiles to avoid dynamic slicing layout issues
    num_val_tiles = val.shape[axis] // tile_shape[axis]
    val_tiles = jnp.split(val, num_val_tiles, axis=axis)
    idx_tiles = jnp.split(idx, num_idx_tiles, axis=axis)

    for val_tile_idx, val_tile in enumerate(val_tiles):
      val_offset = val_tile_idx * tile_shape[axis]
      # Apply to all K blocks
      for idx_tile_idx, idx_tile in enumerate(idx_tiles):
        mask = (idx_tile >= val_offset) & (idx_tile < val_offset + tile_shape[axis])
        gather_tile = jnp.take_along_axis(
            val_tile,
            (idx_tile - val_offset) % tile_shape[axis],
            axis=axis
        )
        accumulators[idx_tile_idx] = jnp.where(mask, gather_tile, accumulators[idx_tile_idx])
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

  
def take_along_axis_refs(values_ref, indices_ref, output_ref, *, axis: int):
  """Gather values by indexing in to all of value with a mask.

  This kernel processes multiple tiles of output (NUM_SUBLANES x K).
  It scans across the entire values_ref (which contains full vocab for the corresponding tokens)
  once, updating all output tiles.
  """
  output_ref[...] = take_along_axis_arrays(values_ref[...], indices_ref[...], axis=axis)
  

@functools.partial(jit, static_argnames=("axis", "interpret",))
def take_along_axis(
    values,
    indices,
    axis,
    interpret: bool = False,
):
  """
  Gather values from `values` array using `indices`.

  Args:
      values: Input values [Batch, VocabSize].
      indices: Indices to gather [Batch, K].
      interpret: Run in interpreter mode (CPU compatible).

  Returns:
      Gathered values: [Batch, K].
  """
  return pl.pallas_call(
      functools.partial(
        take_along_axis_refs,
        axis=axis,
      ),
      out_shape=jax.ShapeDtypeStruct(indices.shape, values.dtype),
      interpret=interpret
  )(values, indices)