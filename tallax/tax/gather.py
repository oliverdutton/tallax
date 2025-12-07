
import functools
import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax.utils import NUM_LANES, NUM_SUBLANES, pad

def pallas_compatible_take_along_axis(val, idx, axis):
  shape = idx.shape
  tile_shape = (NUM_SUBLANES, NUM_LANES)
  val, idx = (pad(x, tile_shape, pad_val=0) for x in (val, idx))
  def _gather(val, idx):
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
        mask = (idx_tile >= idx_offset) & (idx_tile < idx_offset + tile_shape[axis])
        gather_tile = jnp.take_along_axis(
            val_tile,
            idx_tile % tile_shape[axis],
            axis=axis
        )
        i = idx_offset // tile_shape[axis]      
        accumulators[i] = jnp.where(mask, gather_tile, accumulators[i])
    return jnp.concatenate(accumulators, axis=axis)
  batch_axis = 1 - axis
  assert val.shape[batch_axis]==idx.shape[batch_axis]
  return jnp.concatenate(
    [_gather(v, i) 
      for v, i in zip(map(lambda arr: jnp.split(
        arr, arr.shape[batch_axis] // tile_shape[batch_axis], axis=batch_axis), (val, idx)))
    ],
    axis=batch_axis
  )[:shape[0], :shape[1]]

  
def take_along_axis_kernel(values_ref, indices_ref, output_ref, *, axis: int):
  """Gather values by indexing in to all of value with a mask.

  This kernel processes multiple tiles of output (NUM_SUBLANES x K).
  It scans across the entire values_ref (which contains full vocab for the corresponding tokens)
  once, updating all output tiles.
  """
  output_ref[...] = pallas_compatible_take_along_axis(values_ref[...], indices_ref[...], axis=axis)
  

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
        take_along_axis_kernel,
        axis=axis,
      ),
      out_shape=jax.ShapeDtypeStruct(indices.shape, values.dtype),
      interpret=interpret
  )(values, indices)