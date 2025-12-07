
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax.utils import (
    iota_tile,
    NUM_LANES,
    NUM_SUBLANES,
    log2
)

def cumsum_tile(tile, axis):
  n = tile.shape[axis]
  idx = iota_tile(axis)
  for stage in range(log2(n)):
    permutation = idx - 2**stage
    tile += jnp.where(
      permutation>=0,
      jnp.take_along_axis(tile, permutation % n, axis=axis),
      0)
  return tile

def pallas_compatible_cumsum(arr, axis, reverse=False):
  '''
  TPU Pallas lowerable array based implementation of jax.lax.cumsum
  
  Note: most TPU versions do not allow lane sums in bfloat16, so suggest specifying dtype=jnp.float32
  '''
  assert arr.ndim==2
  shape = arr.shape
  tile_shape = (NUM_SUBLANES, NUM_LANES)
  arr = pad(arr, tile_shape)
  def _cumsum(arr):
    n = arr.shape[axis] // tile_shape[axis]
    tiles = jnp.split(arr, n, axis=axis)
    outs = [cumsum_tile(tile, axis) for tile in tiles]
    tile_sums = [tile.sum(axis, keepdims=True) for tile in tiles]
    for i in range(1, n): 
      outs[i] += tile_sums[i-1]
      tile_sums[i] += tile_sums[i-1]
    if reverse:
      # reverse tiles
      out = outs[::-1]
      # reverse within tiles
      reverse_perm = tile_shape[axis] - 1 - iota_tile(axis)
      outs = [jnp.take_along_axis(tile, reversal_perm, axis=axis) for tile in outs]
    return jnp.concatenate(outs, axis=axis)
  
  batch_axis = 1 - axis
  return jnp.concatenate(
    [_cumsum(x, axis=axis) 
      for x in jnp.split(
        arr, arr.shape[batch_axis] // tile_shape[batch_axis], axis=batch_axis)
    ],
    axis=batch_axis
  )[:shape[0], :shape[1]]
