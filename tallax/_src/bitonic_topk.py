"""
Bitonic Top-K for k=NUM_LANES=128 using compressed transpose format.

This implementation is optimized for TPU with k=128 and works entirely in
compressed transpose format to maximize efficiency of permutation operations.

Algorithm:
- Convert input to compressed transpose format: (num_tokens, vocab) -> (NUM_LANES, num_tokens*chunks)
- Build bitonic sequences using stages 1-6 (so sorted in 64 length chunks)
- Cross-tile merge with max selection, reducing tile count
- Progressive sublane permute merging with decreasing distances
- Convert back to original format
"""

import functools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax._src.utils import (
    NUM_LANES,
    NUM_SUBLANES,
    log2,
    flatten,
    ceil_multiple,
    iota_tile,
    pad,
    canonicalize_operand,
    transpose_list_of_lists,
    to_compressed_transpose_format,
    from_compressed_transpose_format,
    to_32bit_dtype,
    join_tiles_to_array,
    split_array_to_tiles,
    
)
from tallax._src.sort import (
    run_compressed_transpose_format_substages_on_tiles,
    compare_and_swap,
    compute_pair_slice_start_index,
)

def legacy_max_arrays(operands, num_keys, axis):
  """Compute max over several operands, sorting using num_keys.

  This function computes the maximum element along the specified axis for multiple
  operands (e.g., values and indices). When comparing elements, it uses the first
  num_keys operands as sort keys to determine which element is "larger".

  Args:
    operands: List of JAX arrays of the same shape
    num_keys: Number of operands to use as sort keys for comparison
    axis: Axis along which to find the maximum (0 or 1)

  Returns:
    List of 1D arrays containing the maximum element for each operand
  """
  if axis == 1:
    # transpose and run on axis 0
    operands = jax.tree.map(lambda x: x.T, operands)
    axis = 0
  assert axis == 0
  unpadded_shape = operands[0].shape
  padded_dim0 = max(2**log2(unpadded_shape[0]), NUM_SUBLANES)
  operands = [pad(x, (padded_dim0, NUM_LANES), val='min') for x in operands]
  
  shape = operands[0].shape
  for _ in range(log2(shape[0] // NUM_SUBLANES)):
    lefts, rights = transpose_list_of_lists([jnp.split(arr,2,axis=0) for arr in operands])
    operands = transpose_list_of_lists(compare_and_swap(lefts, rights, num_keys=num_keys, is_descending=True))[0]
  assert operands[0].shape[0] == NUM_SUBLANES
  assert shape[1] % NUM_LANES == 0

  arrs_tiles = [jnp.split(x, shape[1] // NUM_LANES, axis=1) for x in operands]
  for stage in range(log2(NUM_SUBLANES))[::-1]:  
    permutation = jnp.bitwise_xor(iota_tile(0), 2**stage)
  
    # Apply permutation to all tiles
    arrs_tiles_permuted = jax.tree.map(
      lambda tile: jnp.take_along_axis(tile, permutation, axis=0),
      arrs_tiles
    )
  
    # Compare and merge with permuted values
    outs_tiles = [[] for _ in arrs_tiles]
    for _, (lefts, rights) in enumerate(zip(
          *map(transpose_list_of_lists, (arrs_tiles, arrs_tiles_permuted)),
          strict=True
      )):
        for j, (o, _) in enumerate(compare_and_swap(
            lefts, rights,
            is_descending=True,
            num_keys=num_keys
        )):
          outs_tiles[j].append(o)
    arrs_tiles = outs_tiles
  return [jnp.concatenate(tiles, axis=1)[0,:unpadded_shape[1]] for tiles in arrs_tiles]


def _compute_padded_shape(unpadded_dim0: int, unpadded_dim1: int, k: int) -> tuple[int, int]:
  """Compute padded shape compatible with compressed transpose format requirements.

  This function finds the minimal
  padded shape that satisfies the constraints:
  - dim0 is a power of 2 between NUM_SUBLANES and NUM_LANES (inclusive)
  - dim1 is a multiple of k
  - must be possible to split into tiles so num_elems must be divisible by NUM_SUBLANES * NUM_LANES
  
  Args:
    unpadded_dim0: Original first dimension size
    unpadded_dim1: Original second dimension size

  Returns:
    Tuple of (padded_dim0, padded_dim1) compatible with compressed transpose format
  """
  if unpadded_dim0 >= NUM_LANES:
    dim0 = ceil_multiple(unpadded_dim0, NUM_LANES)
    dim1 = ceil_multiple(unpadded_dim1, max(k, NUM_SUBLANES))
    return (dim0, dim1)

  dim0s = [2**i for i in range(log2(NUM_SUBLANES), log2(NUM_LANES)+1)
    if 2**i >= unpadded_dim0]
  shapes = [
    (dim0, ceil_multiple(unpadded_dim1,
      (max(k, NUM_SUBLANES) * NUM_LANES) // dim0))
    for dim0 in dim0s]
  # take minimal num elements, larger dim0 on ties as cross tile ops are faster than cross lane
  return sorted(shapes, key=lambda x: (x[0] * x[1], -x[0]))[0]

def _max_reduce_bitonic_inter_tile(
    arrs_tiles, *, separation, num_keys: int
):
  """Perform crosstile comparison keeping max values.

  Args:
    arrs_tiles: Tuple of lists of tile arrays
    dim0: First dimension size (padded)
    num_keys: Number of sort keys

  Returns:
    Tuple of lists with half the tiles (max halves only), plus remainder if odd
  """
  num_tiles = len(arrs_tiles[0])
  outs_tiles = [[] for t in arrs_tiles]
  for i in range(num_tiles // 2):
    idx = compute_pair_slice_start_index(i, separation)
    lefts, rights = (
        transpose_list_of_lists(arrs_tiles)[j]
        for j in (idx, idx + separation)
    )
    # Keep only max (left) values, discard min (right)
    for j, (o_left, _) in enumerate(compare_and_swap(
        lefts, rights, is_descending=True, num_keys=num_keys
    )):
      outs_tiles[j].append(o_left)  
  return outs_tiles

def _max_reduce_bitonic_intra_tile(arrs_tiles, *, axis, separation, num_keys):
    # Create permutation indices for tiles using iota_tile
    permutation = jnp.bitwise_xor(iota_tile(axis), separation)
  
    # Apply permutation to all tiles
    arrs_tiles_permuted = jax.tree.map(
      lambda tile: jnp.take_along_axis(tile, permutation, axis=axis),
      arrs_tiles
    )
    # Compare and merge with permuted values
    outs_tiles = [[] for _ in arrs_tiles]
    for _, (lefts, rights) in enumerate(zip(
          *map(transpose_list_of_lists, (arrs_tiles, arrs_tiles_permuted)),
          strict=True
      )):
        for j, (o, _) in enumerate(compare_and_swap(
            lefts, rights,
            is_descending=True,
            num_keys=num_keys
        )):
          outs_tiles[j].append(o)
    return outs_tiles
    
    

# until pl.cdiv(k, NUM_SUBLANES) tiles left. compare at distance ceil_multiple(k, NUM_SUBLANES)
# now the number of tiles is set. 
# then compare cross lane min(log2(pl.cdiv(NUM_LANES, dim0)), num_merges) times. 
# then compare cross sublane log2(pl.cdiv(NUM_SUBLANES, k)) times
def bitonic_topk_arrays(operands: list[jax.Array], k: int = NUM_LANES, num_keys: int = 1, axis: int = 1):
    """
    Progressive bitonic merge for top-k selection.

    Strategy:
    1. Build bitonic sequences (stages 1-6) within tiles
    2. Cross-tile bitonic merge until we reach target tile count
    3. Final progressive merge with lane permutations
    4. Sort final bitonic sequence to descending order

    Args:
        operands: List of JAX arrays of shape (dim0, dim1)
        k: Number of top elements to return (default: NUM_LANES)
        num_keys: Number of sort keys (default: 1)

    Returns:
        List of JAX arrays of shape (original_batch_size, k) with top-k elements
    """
    batch_axis = 1 - axis
    if k > NUM_LANES:
      raise NotImplementedError
    unpadded_k = k
    k = 2**log2(k)
    # Compute padded shape that satisfies alignment requirements
    shape = operands[0].shape
    unpadded_sort_dim = shape[axis]
    if unpadded_k > unpadded_sort_dim:
        raise ValueError
    if axis == 1:
        padded_shape = _compute_padded_shape(*shape, k=k)
    elif axis == 0:
        padded_shape = (
            ceil_multiple(shape[0], max(NUM_SUBLANES, k)),
            ceil_multiple(shape[1], NUM_LANES)
        )
    else:
        raise ValueError
    # Pad both dimensions if needed
    arrs = [pad(op, block_shape=padded_shape, val='min') for op in operands]
    arrs = [x.astype(to_32bit_dtype(x.dtype)) for x in arrs]

    def _max_reduce_bitonic(arrs_tiles, separation, batch_size):
        # separation is comparison distance
        assert separation == 2**log2(separation)
        num_tiles = len(arrs_tiles[0])
        cross_tile = False
        if separation < NUM_SUBLANES:
            # cross sublane
            reduce_fn = functools.partial(_max_reduce_bitonic_intra_tile, axis=0, separation=separation)
        elif separation < num_tiles * NUM_SUBLANES:
            # cross tile
            cross_tile = True
            tile_separation = separation // NUM_SUBLANES
            reduce_fn = functools.partial(_max_reduce_bitonic_inter_tile, separation=tile_separation)
        else:
            # cross lane (due to compressed transpose format)
            lane_separation = batch_size * (separation // (num_tiles * NUM_SUBLANES))
            reduce_fn = functools.partial(_max_reduce_bitonic_intra_tile, axis=1, separation=lane_separation)
        
        # special handling for cross tile as tile to compare to may not exist
        remainder_length = len(arrs_tiles[0]) % (2 * pl.cdiv(k, NUM_SUBLANES))
        if cross_tile and remainder_length:
          remainder_arrs_tiles = [
          x[-remainder_length:] for x in arrs_tiles]
          arrs_tiles = [
          x[:-remainder_length] for x in arrs_tiles]

        arrs_tiles = run_compressed_transpose_format_substages_on_tiles(
          arrs_tiles,
          num_substages=log2(k),
          stage=log2(separation),
          dim0=batch_size,
          num_keys=num_keys,
        )
        arrs_tiles = reduce_fn(arrs_tiles, num_keys=num_keys)

        if cross_tile and remainder_length:
          arrs_tiles = [x + rem for x, rem in zip(arrs_tiles, remainder_arrs_tiles, strict=True)]
        return arrs_tiles

    def _topk_arrays(arrs):
      # Convert to compressed transpose format
      arrs_tiles = jax.tree.map((to_compressed_transpose_format if axis==1 else split_array_to_tiles), arrs)
      batch_size = arrs[0].shape[batch_axis]
      assert batch_size <= NUM_LANES
      log_lanes = log2(NUM_LANES)
      num_tiles = len(arrs_tiles[0])
      num_merges = log2(unpadded_sort_dim) - log2(k)
      num_sublane_merges = log2(pl.cdiv(NUM_SUBLANES, k))
      num_lane_merges = log2(pl.cdiv(unpadded_sort_dim, num_tiles * NUM_SUBLANES))
      num_tile_merges = num_merges - num_sublane_merges - num_lane_merges
      # are intra permutations
  
      # Build bitonic sequences up to length k/2
      for stage in range(1, log2(k)):
        arrs_tiles = run_compressed_transpose_format_substages_on_tiles(
          arrs_tiles,
          num_substages=stage,
          stage=stage,
          dim0=batch_size,
          num_keys=num_keys,
        )
  
      # Progressive merge tiles together as far as possible
      for _ in range(num_tile_merges):
        arrs_tiles = _max_reduce_bitonic(arrs_tiles, separation=ceil_multiple(k, NUM_SUBLANES), batch_size=batch_size)
      num_tiles = len(arrs_tiles[0])
      assert num_tiles == pl.cdiv(k, NUM_SUBLANES), f'{num_tiles=}, should be {pl.cdiv(k, NUM_SUBLANES)}'
      for i in range(num_lane_merges)[::-1]:
        separation = num_tiles * NUM_SUBLANES * 2**i
        arrs_tiles = _max_reduce_bitonic(arrs_tiles, separation=separation, batch_size=batch_size)
      for i in range(num_sublane_merges)[::-1]:
        separation = 2**i
        arrs_tiles = _max_reduce_bitonic(arrs_tiles, separation=separation, batch_size=batch_size)
      # Final sort: convert bitonic sequence to fully descending order
      # Use dim1_offset=k to ensure descending direction
      arrs_tiles = run_compressed_transpose_format_substages_on_tiles(
        arrs_tiles,
        num_substages=log2(k),
        stage=log2(k),
        dim1_offset=k,
        dim0=batch_size,
        num_keys=num_keys,
      )
      return [(from_compressed_transpose_format if axis==1 else join_tiles_to_array)(
        tiles, dim0=(batch_size if axis==1 else ceil_multiple(k, NUM_SUBLANES))) for tiles in arrs_tiles]
    # wrapping to act on batch_size <= NUM_LANES in the kernel 
    arrs = [
      jnp.concatenate(arr_slices, axis=batch_axis)
      for arr_slices in transpose_list_of_lists(
        [_topk_arrays(arrs)
        for arrs in transpose_list_of_lists([
        jnp.split(arr, pl.cdiv(padded_shape[batch_axis], NUM_LANES), axis=batch_axis) for arr in arrs])
    ])]
    return [(arr[:shape[batch_axis],:unpadded_k] if axis==1 else arr[:unpadded_k, :shape[batch_axis]]) for arr in arrs]


def max_arrays(operands, num_keys, axis):
    arrs = bitonic_topk_arrays(operands, num_keys=num_keys, k=1, axis=axis)
    return [x.squeeze(axis) for x in arrs]


def bitonic_topk_refs(
    in_refs,
    out_refs,
    *,
    num_keys: int,
    descending: bool,
    k: int,
):
    """
    Pallas kernel for bitonic top-k with k=128 in compressed transpose format.

    Algorithm:
    1. Pad input to satisfy alignment requirements
    2. Convert to compressed transpose format: (num_tokens, vocab) -> (128, num_tokens*chunks)
    3. Run bitonic top-k stages to select top 128 values per token
    4. Convert back from compressed transpose format
    5. Unpad and extract top-128 per token
    """
    if not descending:
      raise NotImplementedError
    outs = bitonic_topk_arrays(
      [ref[...] for ref in in_refs], k=out_refs[0].shape[1],
      num_keys=num_keys)
    for out, out_ref in zip(outs, out_refs, strict=True):
      out_ref[...] = out.astype(out_ref.dtype)


@functools.partial(
    jit,
    static_argnames=("k", "num_keys", "descending", "interpret"),
)
def bitonic_topk(
    operand: jax.Array | Sequence[jax.Array],
    k: int = NUM_LANES,
    num_keys: int = 1,
    descending: bool = True,
    interpret: bool = False,
) -> tuple[jax.Array, ...]:
    """
    Compute top-k using bitonic sort in compressed transpose format.

    Optimized for k=NUM_LANES=128 only. Works entirely in compressed transpose
    format for maximum TPU efficiency. Supports multiple operands like sort().

    Supports arbitrary input shapes - padding is handled automatically:
    - For small inputs (prod < NUM_LANES2): pads dim0 to make prod = NUM_LANES2
    - For larger inputs: pads both dims minimally to satisfy alignment

    Args:
        operand: Input array(s) of shape [num_tokens, vocab_size].
                Can be a single array or sequence of arrays.
                Any vocab_size is supported (will be padded automatically).
        k: Number of top elements (must be NUM_LANES=128).
        num_keys: Number of arrays to use as sort keys.
        descending: If True, sort in descending order (default for top-k).
        interpret: If True, run in CPU interpret mode.

    Returns:
        Tuple of arrays (same length as input operands):
            - Each array has shape [num_tokens, k]

    Raises:
        ValueError: If k != NUM_LANES
    """
    if k > NUM_LANES:
      raise ValueError(
          f"bitonic_topk only supports k<=NUM_LANES={NUM_LANES}, got k={k}"
      )

    operands, unpadded_shape = canonicalize_operand(operand)
    operands = [pad(x, (NUM_SUBLANES, NUM_LANES),
      val='min' if descending else 'max') for x in operands]
    num_tokens, vocab_size = operands[0].shape
    # Define output shapes
    output_shapes = [
        jax.ShapeDtypeStruct((num_tokens, k), op.dtype)
        for op in operands
    ]
    outputs = pl.pallas_call(
        functools.partial(
            bitonic_topk_refs,
            num_keys=num_keys,
            descending=descending,
            k=k,
        ),
        out_shape=(output_shapes,),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=int(0.9 * 2**27)
        ),
        interpret=interpret,
    )(operands)[0]
    return tuple(x[:unpadded_shape[0], :k] for x in outputs)
