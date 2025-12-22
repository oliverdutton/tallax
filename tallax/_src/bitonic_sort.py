"""
Bitonic Top-K for k<=NUM_LANES (128) using compressed transpose format.

This implementation is optimized for TPU and works entirely in
compressed transpose format to maximize efficiency of permutation operations.

Algorithm:
- Convert input to compressed transpose format: (num_tokens, vocab) -> (NUM_LANES, num_tokens*chunks)
- Build bitonic sequences using stages 1-6 (so sorted in 64 length chunks)
- Cross-tile merge with max selection, reducing tile count
- Progressive sublane permute merging with decreasing distances
- Convert back to original format
"""

PIPELINE_STAGE = 7
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
    create_bit_indicator,
)
from tallax._src.sort import (
    compare_and_swap,
    compute_pair_slice_start_index,
)


def _compute_padded_shape(unpadded_dim0: int, unpadded_dim1: int, k: int) -> tuple[int, int]:
  """Compute padded shape compatible with compressed transpose format requirements.

  This function finds the minimal padded shape that satisfies the constraints:
  - dim0 is a power of 2 between NUM_SUBLANES and NUM_LANES (inclusive)
  - dim1 is a multiple of k
  - num_elems must be divisible by NUM_LANES^2 so mosaic lowers the split and
    concat on full tiles, subtile concat not supported

  Args:
    unpadded_dim0: Original first dimension size
    unpadded_dim1: Original second dimension size
    k: Target top-k size (must be power of 2 for padding calculation purposes)

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
      NUM_LANES * NUM_LANES // dim0))
    for dim0 in dim0s]
  # take minimal num elements, larger dim0 on ties as cross tile ops are faster than cross lane
  return sorted(shapes, key=lambda x: (x[0] * x[1], -x[0]))[0]

### Bitonic Sort Implementation

def _resplit(operands, target_tile_dim0: int):    
    def _resplit_inner(operand):
      tiles = jax.tree.leaves(operand)
      dim0 = tiles[0].shape[0]
      if dim0 == target_tile_dim0:
        return tiles
      elif dim0 > target_tile_dim0:
        return flatten([jnp.split(tile, dim0//target_tile_dim0, axis=0) for tile in tiles])
      else:
        l = target_tile_dim0 // dim0
        return [jnp.concatenate(operand[i*l:(i+1)*l], axis=0) for i in range(len(tiles)//l)]
  
    return [_resplit_inner(x) for x in operands]
   
def _compute_is_descending(stage, tile_start_offset, tile_local_offset, sort_dim_offset, compression_length):
    if type(stage) == int:
        # Stratified optimization based on bit position analysis
        if (stage < log2(NUM_SUBLANES)) or (stage >= log2(compression_length)):
            # Bit only set by iota_tile(0), same pattern for all tiles
            return create_bit_indicator(stage, tile_local_offset + sort_dim_offset)
        else:
            # Bit set by tile_offset, constant within tile, differs across tiles
            return create_bit_indicator(stage, tile_start_offset + sort_dim_offset)

    # tracer stage
    return create_bit_indicator(stage, tile_start_offset + tile_local_offset + sort_dim_offset)

def _bitonic_sort_substage(arrs_tiles, *, substage, stage, num_keys: int, batch_size: int, sort_dim_offset: int = 0, compression_length=None):
    """Perform intra-tile bitonic comparison for sort.

    Args:
      arrs_tiles: Tuple of lists of tile arrays
      axis: Axis along which to apply permutation (0 or 1)
      separation: Distance between elements to compare within tile
      stage: Current sorting stage
      num_keys: Number of sort keys
      sort_dim_offset: Offset for bitonic order calculation
      batch_size: Batch size for computing tile offsets

    Returns:
      Tuple of lists of tiles with updated values
    """
    separation = 2**substage
    if type(arrs_tiles[0])!=list:
      # if still arrays, we make it into one big tile so its sanitized to list[list[jax.ndarray]]
      arrs_tiles = jax.tree.map(jax.tree.leaves, arrs_tiles)
    if compression_length is None:
      compression_length = len(arrs_tiles[0]) * arrs_tiles[0][0].shape[0]
    if separation < NUM_SUBLANES or separation >= compression_length:
      # we need to permute within tiles
      axis = int(separation < NUM_SUBLANES)
      intra_tile_separation = separation if axis==0 else (separation // compression_length)
      
      # we need hardware tiles to lower the permute
      arrs_tiles = _resplit(arrs_tiles, NUM_SUBLANES)
      # Compute is_descending for each tile based on bitonic pattern
      tile_local_offset = iota_tile(0) + (iota_tile(1) // batch_size) * compression_length
      is_right_half = create_bit_indicator(log2(intra_tile_separation), iota_tile(axis))
      permutation = jnp.bitwise_xor(iota_tile(axis), intra_tile_separation)
      # Apply permutation to all tiles
      arrs_tiles_permuted = jax.tree.map(
        lambda tile: jnp.take_along_axis(tile, permutation, axis=axis),
        arrs_tiles
      )
  
      # Compare and merge with permuted values
      outs_tiles = [[None for _ in t] for t in arrs_tiles]
      for idx, (lefts, rights) in enumerate(zip(
            *map(transpose_list_of_lists, (arrs_tiles, arrs_tiles_permuted)),
            strict=True
        )):
          for arr_idx, out in enumerate(compare_and_swap(
            lefts, rights,
            is_descending=      _compute_is_descending(
              stage=stage, 
              tile_start_offset=idx*NUM_SUBLANES, 
              tile_local_offset=tile_local_offset, 
              sort_dim_offset=sort_dim_offset, 
              compression_length=compression_length
            ),
            is_right_half=is_right_half,
            num_keys=num_keys
          )):
            outs_tiles[arr_idx][idx] = out
    else:
      # Comparison between tiles
      arrs_tiles = _resplit(arrs_tiles, separation)
      tile_shape = arrs_tiles[0][0].shape      
      num_tiles = len(arrs_tiles[0])
      tile_separation = separation // tile_shape[0]
      
      tile_local_offset = iota_tile(0, tile_shape) + (iota_tile(1, tile_shape) // batch_size) * compression_length

      outs_tiles = [[None for _ in t] for t in arrs_tiles]
      for i in range(num_tiles // 2):
        idx = compute_pair_slice_start_index(i, separation=tile_separation)
        lefts, rights = (transpose_list_of_lists(arrs_tiles)[j] for j in (idx, idx + tile_separation))
        for arr_idx, (out_left, out_right) in enumerate(compare_and_swap(
            lefts, rights, is_descending=_compute_is_descending(
              stage=stage, 
              tile_start_offset=idx*tile_shape[0], 
              tile_local_offset=tile_local_offset, 
              sort_dim_offset=sort_dim_offset, 
              compression_length=compression_length
            ),
            num_keys=num_keys
        )):
          outs_tiles[arr_idx][idx] = out_left
          outs_tiles[arr_idx][idx + tile_separation] = out_right
    assert all(not any([v is None for v in out_tiles]) for out_tiles in outs_tiles)
    return outs_tiles


def bitonic_sort_arrays(operands: list[jax.Array], num_keys: int = 1, axis: int = 1, descending: bool = False):
    """
    Bitonic sort using compressed transpose format with full tile unrolling.

    Similar to bitonic_topk_arrays but performs full sort without reduction.
    Uses the same tiling strategy and format conversion for efficient TPU execution.

    Handles arbitrary sort dimensions efficiently:
    - Dimensions ≤ NUM_LANES (128): Uses compressed transpose format substages
    - Dimensions > NUM_LANES: Extends with cross-lane permutation substages
    - Example: (8, 2048) sorted using stage-based bitonic reduce with full tile unrolling

    Args:
        operands: List of JAX arrays of shape (dim0, dim1)
        num_keys: Number of sort keys (default: 1)
        axis: Axis along which to perform sort (0 or 1)
        descending: If True, sort in descending order

    Returns:
        List of JAX arrays of same shape as input, sorted along specified axis
    """
    batch_axis = 1 - axis
    shape = operands[0].shape
    #unpadded_sort_dim = shape[axis]

    if axis == 1:
        padded_shape = _compute_padded_shape(shape[0], shape[1], k=NUM_SUBLANES)
    elif axis == 0:
        padded_shape = (
            ceil_multiple(shape[0], NUM_SUBLANES),
            ceil_multiple(shape[1], NUM_LANES)
        )
    else:
        raise ValueError

    # Pad both dimensions if needed
    # For ascending sort, pad with 'max' so padding values sort to the end
    # For descending sort, pad with 'min' so padding values sort to the end
    arrs = [pad(op, block_shape=padded_shape, val='min' if descending else 'max') for op in operands]
    arrs = [x.astype(to_32bit_dtype(x.dtype)) for x in arrs]

    def _sort_arrays(arrs):
      # Convert to compressed transpose format
      arrs_tiles = jax.tree.map((to_compressed_transpose_format if axis==1 else split_array_to_tiles), arrs)
      batch_size = arrs[0].shape[batch_axis]
      assert batch_size <= NUM_LANES
      sort_dim = arrs[0].shape[axis]
      num_stages = log2(sort_dim)

      # Offset to control ascending vs descending final order
      sort_dim_offset = int(descending) * sort_dim

      # Run all bitonic sort stages
      compression_length = arrs_tiles[0].shape[0]
      out_arrs_tiles = []
      l = 2**PIPELINE_STAGE
      for i in range(compression_length // l):
        arrs_tiles_ = [arr[i*l:(i+1)*l] for arr in arrs_tiles]
        for stage in range(1, PIPELINE_STAGE+1):
          for substage in range(stage)[::-1]:
            arrs_tiles_ = _bitonic_sort_substage(arrs_tiles_, substage=substage, stage=stage, num_keys=num_keys, batch_size=batch_size, sort_dim_offset=sort_dim_offset+i*l, compression_length = compression_length)
        out_arrs_tiles.append([jnp.concat(x, axis=0) for x in arrs_tiles_])
      arrs_tiles = transpose_list_of_lists(out_arrs_tiles)
      
      for stage in range(PIPELINE_STAGE+1, num_stages + 1):
        for substage in range(stage)[::-1]:
          arrs_tiles = _bitonic_sort_substage(arrs_tiles, substage=substage, stage=stage, num_keys=num_keys, batch_size=batch_size, sort_dim_offset=sort_dim_offset, compression_length = compression_length)

      # Convert back from compressed transpose format
      if axis == 1:
        arrs = [from_compressed_transpose_format(tiles, dim0=batch_size) for tiles in arrs_tiles]
      else:
        arrs = [join_tiles_to_array(tiles, dim0=ceil_multiple(sort_dim, NUM_SUBLANES)) for tiles in arrs_tiles]
      return arrs

    # wrapping to act on batch_size <= NUM_LANES in the kernel
    arrs = [
      jnp.concatenate(arr_slices, axis=batch_axis)
      for arr_slices in transpose_list_of_lists(
        [_sort_arrays(arrs)
        for arrs in transpose_list_of_lists([
        jnp.split(arr, pl.cdiv(padded_shape[batch_axis], NUM_LANES), axis=batch_axis) for arr in arrs])
    ])]
    # Unpad to original shape
    return [arr[:shape[0], :shape[1]] for arr in arrs]


def bitonic_sort_refs(
    in_refs,
    out_refs,
    *,
    num_keys: int,
    descending: bool,
):
    """
    Pallas kernel for bitonic sort in compressed transpose format.

    Args:
        in_refs: Input array references
        out_refs: Output array references
        num_keys: Number of sort keys
        descending: Sort in descending order
    """
    outs = bitonic_sort_arrays(
      [ref[...] for ref in in_refs],
      num_keys=num_keys,
      descending=descending,
    )
    for out, out_ref in zip(outs, out_refs, strict=True):
      out_ref[...] = out.astype(out_ref.dtype)


@functools.partial(
    jit,
    static_argnames=("num_keys", "descending", "interpret"),
)
def bitonic_sort(
    operand: jax.Array | Sequence[jax.Array],
    num_keys: int = 1,
    descending: bool = False,
    interpret: bool = False,
) -> tuple[jax.Array, ...]:
    """
    Sort arrays using bitonic sort in compressed transpose format.

    Optimized for sorting power-of-2 sized arrays on TPU. Works entirely in
    compressed transpose format for maximum efficiency. Similar to bitonic_topk
    but performs full sort.

    Supports arbitrary input shapes - padding is handled automatically to
    nearest power of 2.

    Args:
        operand: Input array(s) of shape [batch, sort_dim].
                Can be a single array or sequence of arrays.
                Any sort_dim is supported (will be padded automatically).
        num_keys: Number of arrays to use as sort keys.
        descending: If True, sort in descending order.
        interpret: If True, run in CPU interpret mode.

    Returns:
        Tuple of arrays (same length as input operands):
            - Each array has same shape as input, sorted along axis 1

    Example:
        >>> import jax.numpy as jnp
        >>> from tallax._src.bitonic_topk import bitonic_sort
        >>> x = jnp.array([[3, 1, 4, 2], [8, 5, 7, 6]], dtype=jnp.int32)
        >>> result = bitonic_sort(x, descending=False)
        >>> print(result[0])
        [[1 2 3 4]
         [5 6 7 8]]
    """
    operands, unpadded_shape = canonicalize_operand(operand)
    # For ascending sort, pad with 'max' so padding values sort to the end
    # For descending sort, pad with 'min' so padding values sort to the end
    operands = [pad(x, (NUM_SUBLANES, NUM_LANES),
      val='min' if descending else 'max') for x in operands]
    batch_size, sort_dim = operands[0].shape

    # Define output shapes
    output_shapes = [
        jax.ShapeDtypeStruct((batch_size, sort_dim), op.dtype)
        for op in operands
    ]
    outputs = pl.pallas_call(
        functools.partial(
            bitonic_sort_refs,
            num_keys=num_keys,
            descending=descending,
        ),
        out_shape=(output_shapes,),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=int(0.9 * 2**27)
        ),
        interpret=interpret,
    )(operands)[0]
    return tuple(x[:unpadded_shape[0], :unpadded_shape[1]] for x in outputs)

'''
def _run_bitonic_stages_on_transpose_refs(
    transpose_refs,
    *,
    batch_size: int,
    sort_dim: int,
    num_keys: int,
    descending: bool,
    unroll: int = 128,
):
    """Execute bitonic sort stages on data in compressed transpose format.

    This function operates on transpose_refs which hold data in compressed
    transpose format throughout the sorting process to minimize conversions.

    Args:
        transpose_refs: References to arrays in compressed transpose format
                       Shape: (sort_dim * batch_size / NUM_LANES, NUM_LANES)
        batch_size: Original batch size (dim0 before transposing)
        sort_dim: Original sort dimension (dim1 before transposing)
        num_keys: Number of sort keys
        descending: Sort in descending order
        unroll: Number of tiles to allow in dim0
    """
    # transpose_refs shape: (transpose_dim0, NUM_LANES)
    # where transpose_dim0 = (sort_dim * batch_size) / NUM_LANES
    transpose_dim0 = transpose_refs[0].shape[0]
    num_tiles = transpose_dim0 // NUM_SUBLANES
    num_stages = log2(sort_dim)

    # Offset to control ascending vs descending final order
    sort_dim_offset = int(descending) * sort_dim

    # Read transpose refs into tiles
    arrs = jax.tree.leaves([ref[...] for ref in transpose_refs])
    arrs_tiles = [split_array_to_tiles(arr) for arr in arrs]

    # Stages 1 to log2(unroll*NUM_SUBLANES) - fully unrolled
    num_unrolled_stages = min(num_stages, log2(unroll * NUM_SUBLANES))

    for stage in range(1, num_unrolled_stages + 1):
        arrs_tiles = _run_bitonic_stage_on_tiles(
            arrs_tiles,
            stage=stage,
            batch_size=batch_size,
            num_keys=num_keys,
            sort_dim_offset=sort_dim_offset,
            sort_dim=sort_dim
        )

    # Write back unrolled results to transpose refs
    arrs = [jnp.concatenate(tiles, axis=0) for tiles in arrs_tiles]
    for ref, arr in zip(transpose_refs, arrs, strict=True):
        ref[...] = arr

    # Dynamic stages beyond unrolled stages
    if num_stages > num_unrolled_stages:
        max_substage = log2(num_tiles * NUM_SUBLANES)
        tile_local_offset = iota_tile(0) + (iota_tile(1) // batch_size) * num_tiles * NUM_SUBLANES

        @pl.loop(num_unrolled_stages + 1, num_stages + 1)
        def run_dynamic_stage(stage):
            # Substages for stage s (1-indexed) are: s-1, s-2, ..., 0 (0-indexed)
            # Cross-lane substages (>= max_substage) - conditional
            # Note: stage is 1-indexed, so stage-1 is the highest 0-indexed substage needed
            for substage_0indexed in range(max_substage, num_stages)[::-1]:
                # Run this substage if stage > substage_0indexed+1 (converting back to 1-indexed stage)
                @pl.when(stage > substage_0indexed)
                def run_cross_lane_substage():
                    # Read from refs and split into standard tiles
                    ru[ref[...] for ref in transpose_refs])
                    
                    # Write back to refs
                    arrs = [jnp.concatenate(tiles, axis=0) for tiles in outs_tiles]
                    for ref, arr in zip(transpose_refs, arrs, strict=True):
                        ref[...] = arr

            # Compressed format substages (0 to max_substage-1) - all run unconditionally
            # These always run because stage >= num_unrolled_stages + 1
            for substage_0indexed in range(0, max_substage)[::-1]:
                # Read from refs and split into standard tiles
                arrs = jax.tree.leaves([ref[...] for ref in transpose_refs])
                arrs_tiles = [_resplit(arr, NUM_SUBLANES) for arr in arrs]

                arrs_tiles = _run_compressed_transpose_format_substage_on_tiles(
                    arrs_tiles,
                    substage=substage_0indexed,
                    stage=stage,
                    sort_dim_offset=sort_dim_offset,
                    batch_size=batch_size,
                    num_keys=num_keys,
                )

                # Write back results
                arrs = [jnp.concatenate(tiles, axis=0) for tiles in arrs_tiles]
                for ref, arr in zip(transpose_refs, arrs, strict=True):
                    ref[...] = arr


def bitonic_sort_refs(
    in_refs,
    out_refs,
    transpose_scratch_refs,
    *,
    num_keys: int,
    descending: bool,
    unroll: int = 128,
):
    """
    Pallas kernel for bitonic sort with ref slicing optimization.

    This implementation uses ref slicing to reduce compile times by:
    1. Reading input refs and converting to compressed transpose format
    2. Keeping data in compressed format throughout sorting in transpose_scratch_refs
    3. Converting back to normal format only at the end

    Args:
        in_refs: Input array references
        out_refs: Output array references
        transpose_scratch_refs: Scratch refs for compressed transpose format data
        num_keys: Number of sort keys
        descending: Sort in descending order
        unroll: Number of tiles to allow in dim0
    """
    shape = in_refs[0].shape
    batch_size = shape[0]
    sort_dim = shape[1]

    # Verify sort_dim is a power of 2 and compatible with compressed transpose format
    assert sort_dim % NUM_SUBLANES == 0, \
        f"sort_dim {sort_dim} must be divisible by NUM_SUBLANES {NUM_SUBLANES}"
    assert sort_dim == 2**log2(sort_dim), \
        f"sort_dim {sort_dim} must be a power of 2"

    # Read input refs, convert to compressed transpose format, write to transpose_scratch_refs
    arrs = [ref[...] for ref in in_refs]
    arrs = [x.astype(to_32bit_dtype(x.dtype)) for x in arrs]
    arrs_tiles = jax.tree.map(to_compressed_transpose_format, arrs)
    # In compressed transpose format, tiles are concatenated along dim0 -> (sort_dim, NUM_LANES)
    arrs_transposed = [jnp.concatenate(tiles, axis=0) for tiles in arrs_tiles]

    for ref, arr in zip(transpose_scratch_refs, arrs_transposed, strict=True):
        ref[...] = arr

    # Run all sorting stages on transpose_scratch_refs
    _run_bitonic_stages_on_transpose_refs(
        transpose_scratch_refs,
        batch_size=batch_size,
        sort_dim=sort_dim,
        num_keys=num_keys,
        descending=descending,
        unroll=unroll,
    )

    # Read transpose refs, decompress and transpose, write to output refs
    arrs_transposed = [ref[...] for ref in transpose_scratch_refs]
    arrs_tiles = jax.tree.map(split_array_to_tiles, arrs_transposed)
    arrs = [from_compressed_transpose_format(tiles, dim0=batch_size) for tiles in arrs_tiles]

    for ref, arr in zip(out_refs, arrs, strict=True):
        ref[...] = arr.astype(ref.dtype)
'''