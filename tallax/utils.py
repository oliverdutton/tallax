
import math
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl


# TPU hardware constants
NUM_SUBLANES = 8
NUM_LANES = 128

def is_cpu_platform():
  return jax.default_backend() == "cpu"

def _log2(x: int) -> int:
  """Returns ceiling of log2(x)."""
  return math.ceil(math.log2(x))


def _max_int(a, b):
  """Max of two values, accepts both static and dynamic ints."""
  if not all(map(lambda v: type(v) == int, (a, b))):
    return jnp.maximum(a, b)
  return max(a, b)


def _all_concrete_ints(*args):
  """Check if all arguments are concrete Python integers."""
  return all(map(lambda v: type(v) == int, args))


def _get_dtype_info(x):
  """Get finfo or iinfo for array dtype."""
  dtype = x.dtype
  if jnp.issubdtype(dtype, jnp.floating):
    return jnp.finfo(x)
  elif jnp.issubdtype(dtype, jnp.integer):
    return jnp.iinfo(x)
  else:
    raise ValueError('Only int and float supported')


def _pad(x, descending=False):
  """Pad array to satisfy alignment requirements.

  Pads to multiple of NUM_SUBLANES in dim0 and power of 2 in dim1.
  Uses max value (or nan) as padding to not affect sort order.
  """
  dim0, dim1 = x.shape
  pad_dim0 = pl.cdiv(dim0, NUM_SUBLANES) * NUM_SUBLANES
  pad_dim1 = max(2**_log2(dim1), NUM_LANES)

  pad_val = _get_dtype_info(x).max
  if jnp.issubdtype(x.dtype, jnp.floating):
    pad_val = jnp.nan

  return jnp.pad(
      x,
      ((0, pad_dim0 - dim0),
       (0, pad_dim1 - dim1) if not descending else (pad_dim1 - dim1, 0)),
      mode='constant',
      constant_values=pad_val
  )


def _standardize(x):
  """Standardize float values for sorting.

  Converts NaNs to a specific value and normalizes +/-0.
  """
  nan_val = _sortable_int_to_float(jnp.iinfo(jnp.int32).max - 1)
  x = jnp.where(jnp.isnan(x), nan_val, x)
  x = jnp.where(x == 0, 0, x)
  return x


def _is_32bit(x):
  """Check if array has 32-bit dtype."""
  return x.dtype.itemsize == 4


def _to_32bit_dtype(operand_dtype):
  """Convert dtype to corresponding 32-bit dtype."""
  for dtype_class, dtype_32bit in {
      jnp.floating: jnp.float32,
      jnp.integer: jnp.int32,
      jnp.bool_: jnp.int32
  }.items():
    if jnp.issubdtype(operand_dtype, dtype_class):
      return dtype_32bit
  raise ValueError('dtype not recognized')


def _same_shape_dtype(ref1, ref2):
  """Check if two refs have same shape and dtype."""
  return (ref1.dtype == ref2.dtype) and (ref1.shape == ref2.shape)


def _canonicalize_operand(operand):
  """Convert operand to list of arrays and validate shapes."""
  operands = jax.tree.leaves(operand)
  shapes = [x.shape for x in operands]
  if len(set(shapes)) != 1:
    raise ValueError(f'Inputs must all have the same shape, but found {shapes=}')
  shape = shapes[0]
  if len(shape) != 2:
    raise ValueError('Only 2D inputs supported')
  return operands, shape


### Float-Int Conversion for Sortable Representation

def _float_to_sortable_int(x: jnp.ndarray, standardize=True) -> jnp.ndarray:
  """Transform float32 bits into sortable int32 representation.

  Positive floats map to [INT_MIN, -1].
  Negative floats map to [INT_MAX, 0] with reversed order.
  """
  if standardize:
    x = _standardize(x)
  i = x.view(jnp.int32)
  return jnp.where(i < 0, i ^ 0x7FFFFFFF, i)


def _sortable_int_to_float(i: jnp.ndarray) -> jnp.ndarray:
  """Inverse transformation from sortable int32 back to float32."""
  return jnp.where(i < 0, i ^ 0x7FFFFFFF, i).view(jnp.float32)


### BF16-U16 Packing for Optimization

def _pack_bf16_u16_to_i32(val, index):
  """Pack bfloat16 value and uint16 index into single int32.

  BF16 in F32 has empty lower 16 bits where we pack the index.
  This allows sorting while preserving original indices.
  """
  assert index.dtype == jnp.int32
  val_f32 = _standardize(val.astype(jnp.float32))
  index = jnp.where(val_f32 < 0, index.shape[1] - index, index)
  return _float_to_sortable_int(
      ((val_f32.view(jnp.int32) & ~0xFFFF) | index).view(jnp.float32),
      standardize=False
  )


def _unpack_bf16_u16_from_i32(packed):
  """Extract original bfloat16 value and uint16 index from packed int32."""
  assert packed.dtype == jnp.int32, f'found {packed.dtype}'
  packed = _sortable_int_to_float(packed)
  val = (packed.view(jnp.int32) & ~0xFFFF).view(jnp.float32).astype(jnp.bfloat16)
  index = packed.view(jnp.int32) & 0xFFFF
  index = jnp.where(val < 0, index.shape[1] - index, index)
  return val, index


### Tile Operations

def _split_array_to_tiles(arr):
  """Split 2D array into flat list of (NUM_SUBLANES, NUM_LANES) tiles."""
  num_rows, num_cols = arr.shape
  tile_rows = num_rows // NUM_SUBLANES
  tile_cols = num_cols // NUM_LANES

  tiles = []
  for row in range(tile_rows):
    for col in range(tile_cols):
      tile = arr[
          row * NUM_SUBLANES: (row + 1) * NUM_SUBLANES,
          col * NUM_LANES: (col + 1) * NUM_LANES,
      ]
      tiles.append(tile)
  return tiles


def _join_tiles_to_array(target_shape, tiles):
  """Reconstruct 2D array from flat list of tiles."""
  num_rows, num_cols = target_shape
  tile_rows, tile_cols = tiles[0].shape
  grid_cols = num_cols // tile_cols

  rows = []
  for i in range(len(tiles) // grid_cols):
    row_tiles = tiles[i * grid_cols: (i + 1) * grid_cols]
    rows.append(jnp.concatenate(row_tiles, axis=-1))

  return jnp.concatenate(rows, axis=-2)


def _iota_tile(dim):
  """Create iota array with tile shape."""
  return lax.broadcasted_iota(jnp.int32, (NUM_SUBLANES, NUM_LANES), dim)


def _convert_to_sublane_sort_format(arr):
  """Convert array to sublane-oriented format for faster permutes."""
  arrs = [
      arr[:, i * NUM_LANES:(i + 1) * NUM_LANES]
      for i in range(pl.cdiv(arr.shape[1], NUM_LANES))
  ]
  arr = jnp.concatenate(arrs, axis=0).T # (128, n*b)
  if arr.shape[1] < NUM_LANES:
    arr = _pad(arr)
  tiles = _split_array_to_tiles(arr)
  return tiles


def _convert_from_sublane_sort_format(tiles, shape):
  """Convert from sublane format back to original layout."""
  b, m = shape
  assert m >= NUM_LANES
  n = m // NUM_LANES
  dim1 = len(tiles) * NUM_SUBLANES
  arr = _join_tiles_to_array((NUM_LANES, dim1), tiles) # (128, n*b)
  if dim1 != n * b:
    arr = arr[..., :n * b]
  arr = arr.T
  return jnp.concatenate(
      [arr[i * b:(i + 1) * b] for i in range(arr.shape[0] // b)],
      axis=1
  )


### Gather Operations

_gather_sublane = lambda x, index: jax.lax.gather(
    x, index[..., None],
    jax.lax.GatherDimensionNumbers(
        offset_dims=(),
        collapsed_slice_dims=(0,),
        start_index_map=(0,),
        operand_batching_dims=(1,),
        start_indices_batching_dims=(1,),
    ),
    slice_sizes=(1, 1)
)

_gather_lane = jax.vmap(lambda x, index: x[index])


### Loop Utilities

def _unrolled_fori_loop(length: int, body_fn, init_val, unroll: int):
  """Execute for loop with manual unrolling for better performance."""
  unroll = min(length, unroll)

  def unrolled_body(i, carry):
    i *= unroll
    for j in range(unroll):
      carry = body_fn(i + j, carry)
    return carry

  carry = jax.lax.fori_loop(0, length // unroll, unrolled_body, init_val)
  for j in range(length % unroll):
    carry = body_fn((length // unroll) * unroll + j, carry)
  return carry


def _transpose_list_of_lists(tree):
  """Transpose nested list structure."""
  outer = jax.tree.structure(type(tree)('*') * len(tree))
  inner = jax.tree.structure(type(tree[0])('*') * len(tree[0]))
  return jax.tree.transpose(outer, inner, tree)
