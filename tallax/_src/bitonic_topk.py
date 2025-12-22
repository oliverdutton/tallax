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
    run_compressed_transpose_format_substages_on_tiles,
    compare_and_swap,
    compute_pair_slice_start_index,
    _run_compressed_transpose_format_substage_on_tiles,
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

def _max_reduce_bitonic_inter_tile(
    arrs_tiles, *, separation, num_keys: int
):
  """Perform crosstile comparison keeping max values.

  Args:
    arrs_tiles: Tuple of lists of tile arrays
    separation: Distance between tiles to compare
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
    """Perform intra-tile comparison keeping max values.

    Args:
      arrs_tiles: Tuple of lists of tile arrays
      axis: Axis along which to apply permutation (0 or 1)
      separation: Distance between elements to compare within tile
      num_keys: Number of sort keys

    Returns:
      Tuple of lists of tiles with updated values
    """
    # Create permutation indices for tiles using iota_tile
    permutation = jnp.bitwise_xor(iota_tile(axis), separation)
    is_right_half = create_bit_indicator(log2(separation), iota_tile(axis))

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
            is_descending=True,
            is_right_half=is_right_half,
            num_keys=num_keys
        )):
          outs_tiles[arr_idx][idx] = out
    assert all(not any([v is None for v in out_tiles]) for out_tiles in outs_tiles)
    return outs_tiles
    
    

# until pl.cdiv(k, NUM_SUBLANES) tiles left. compare at distance ceil_multiple(k, NUM_SUBLANES)
# now the number of tiles is set. 
# then compare cross lane min(log2(pl.cdiv(NUM_LANES, dim0)), num_merges) times. 
# then compare cross sublane log2(pl.cdiv(NUM_SUBLANES, k)) times
def bitonic_topk_arrays(operands: list[jax.Array], k: int = NUM_LANES, num_keys: int = 1, axis: int = 1, min_padded_dim0: int | None = None):
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
        axis: Axis along which to perform top-k (0 or 1)
        min_padded_dim0: Can be used to tradeoff ALU vs lane permute intensity.
          E.g. (8, 2048) can be put into compressed format of 16 (8, 128) tiles
          which induces 4 lane permute ops at the end which have high latency.
          Alternatively padding to (128, 2048) leads to an uncompressed transpose
          of 256 (8, 128) tiles and avoids lane permutes but greatly increases
          ALU work. Can be tuned.

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
        if min_padded_dim0 is None:
            min_padded_dim0 = shape[0]
        padded_shape = _compute_padded_shape(min_padded_dim0, shape[1], k=k)
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
          batch_size=batch_size,
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
          batch_size=batch_size,
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
        separation = k * (2**i)
        arrs_tiles = _max_reduce_bitonic(arrs_tiles, separation=separation, batch_size=batch_size)
      # Final sort: convert bitonic sequence to fully descending order
      # Use sort_dim_offset=k to ensure descending direction
      arrs_tiles = run_compressed_transpose_format_substages_on_tiles(
        arrs_tiles,
        num_substages=log2(k),
        stage=log2(k),
        sort_dim_offset=k,
        batch_size=batch_size,
        num_keys=num_keys,
      )
      
      arrs = [join_tiles_to_array(
        tiles, dim0=ceil_multiple(k, NUM_SUBLANES)) for tiles in arrs_tiles]
      if axis == 1:
        arrs = [x.T for x in arrs]
      return arrs
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


### Bitonic Sort Implementation

def _bitonic_reduce_inter_tile(
    arrs_tiles, *, separation, stage, num_keys: int, sort_dim_offset: int = 0
):
  """Perform cross-tile bitonic comparison for sort.

  Unlike _max_reduce_bitonic_inter_tile, this keeps both halves (no reduction).

  Args:
    arrs_tiles: Tuple of lists of tile arrays
    separation: Distance between tiles to compare
    stage: Current sorting stage
    num_keys: Number of sort keys
    sort_dim_offset: Offset for bitonic order calculation

  Returns:
    Tuple of lists with same number of tiles (both halves kept)
  """
  num_tiles = len(arrs_tiles[0])
  outs_tiles = [[None for _ in range(num_tiles)] for t in arrs_tiles]
  for i in range(num_tiles // 2):
    idx = compute_pair_slice_start_index(i, separation)
    tile_offset_left = idx * NUM_SUBLANES

    lefts, rights = (
        transpose_list_of_lists(arrs_tiles)[j]
        for j in (idx, idx + separation)
    )

    # Compute is_descending based on bitonic pattern
    is_descending = create_bit_indicator(stage, sort_dim_offset + tile_offset_left)

    # Keep both halves (no reduction), preserving tile order
    for j, (o_left, o_right) in enumerate(compare_and_swap(
        lefts, rights, is_descending=is_descending, num_keys=num_keys
    )):
      outs_tiles[j][idx] = o_left
      outs_tiles[j][idx + separation] = o_right

  assert all(not any([v is None for v in out_tiles]) for out_tiles in outs_tiles)
  return outs_tiles


def _bitonic_reduce_intra_tile(arrs_tiles, *, axis, separation, stage, num_keys, sort_dim_offset: int = 0, batch_size: int):
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
    # Create permutation indices for tiles using iota_tile
    permutation = jnp.bitwise_xor(iota_tile(axis), separation)
    is_right_half = create_bit_indicator(log2(separation), iota_tile(axis))

    # Apply permutation to all tiles
    arrs_tiles_permuted = jax.tree.map(
      lambda tile: jnp.take_along_axis(tile, permutation, axis=axis),
      arrs_tiles
    )

    # Compute is_descending for each tile based on bitonic pattern
    num_tiles = len(arrs_tiles[0])
    tile_local_offset = iota_tile(0) + (iota_tile(1) // batch_size) * num_tiles * NUM_SUBLANES

    def compute_is_descending(idx):
      if axis == 0:
        # Cross-sublane
        tile_offset = idx * NUM_SUBLANES
        return create_bit_indicator(stage, sort_dim_offset + tile_offset + tile_local_offset)
      else:
        # Cross-lane
        lane_offset = idx * NUM_SUBLANES + (iota_tile(1) // batch_size) * num_tiles * NUM_SUBLANES
        return create_bit_indicator(stage, sort_dim_offset + lane_offset * batch_size)

    # Compare and merge with permuted values
    outs_tiles = [[None for _ in t] for t in arrs_tiles]
    for idx, (lefts, rights) in enumerate(zip(
          *map(transpose_list_of_lists, (arrs_tiles, arrs_tiles_permuted)),
          strict=True
      )):
        for arr_idx, out in enumerate(compare_and_swap(
            lefts, rights,
            is_descending=compute_is_descending(idx),
            is_right_half=is_right_half,
            num_keys=num_keys
        )):
          outs_tiles[arr_idx][idx] = out
    assert all(not any([v is None for v in out_tiles]) for out_tiles in outs_tiles)
    return outs_tiles


def _resplit(arrs_or_tiles, target_tile_dim0):
    """Resplit arrays or tiles to have target dim0 size.

    Polymorphic function that handles:
    - Single array: (total_dim0, NUM_LANES) -> list of tiles
    - List of tiles: concatenates then re-splits
    - List of lists: applies recursively to each sublist

    Args:
        arrs_or_tiles: Array, list of arrays, or nested list
        target_tile_dim0: Target dim0 size for each tile

    Returns:
        List of tiles (or list of lists if input was nested)
    """
    # Base case: single array
    if isinstance(arrs_or_tiles, jnp.ndarray):
        arr = arrs_or_tiles
        total_dim0 = arr.shape[0]
        assert total_dim0 % target_tile_dim0 == 0, \
            f"Array dim0 {total_dim0} not divisible by target_tile_dim0 {target_tile_dim0}"
        num_tiles = total_dim0 // target_tile_dim0
        return list(jnp.split(arr, num_tiles, axis=0))

    # Recursive case: list
    if isinstance(arrs_or_tiles, (list, tuple)):
        # Check if it's a list of arrays (all elements are arrays)
        if all(isinstance(x, jnp.ndarray) for x in arrs_or_tiles):
            # List of tiles - concatenate and re-split
            arr = jnp.concatenate(arrs_or_tiles, axis=0)
            return _resplit(arr, target_tile_dim0)
        else:
            # List of lists - apply recursively
            return [_resplit(x, target_tile_dim0) for x in arrs_or_tiles]

    raise ValueError(f"Unsupported type: {type(arrs_or_tiles)}")


def _compute_is_descending_for_tile(stage, tile_idx, batch_size, num_tiles, sort_dim_offset, tile_local_offset, sort_dim):
    """Compute is_descending for a tile with stratified optimizations.

    Optimizes using clear stratification rules similar to sort.py:
    1. stage < log2(NUM_SUBLANES): Pattern same across all tiles (only sublane varies)
    2. stage < log2(num_tiles * NUM_SUBLANES): Scalar per tile (tiles differ)
    3. stage < log2(sort_dim): Pattern same across all tiles (tile_offset insignificant)
    4. stage >= log2(sort_dim): Global scalar (final stage, bit never set)

    Args:
        stage: Current sorting stage
        tile_idx: Index of the current tile
        batch_size: Batch size
        num_tiles: Total number of tiles
        sort_dim_offset: Offset for bitonic order calculation
        tile_local_offset: Precomputed tile local offset array
        sort_dim: Size of dimension being sorted

    Returns:
        is_descending value (scalar or array)
    """
    tile_offset = tile_idx * NUM_SUBLANES

    if type(stage) == int:
        # Stratified optimization based on bit position analysis
        if stage < log2(NUM_SUBLANES):
            # Bit only set by iota_tile(0), same pattern for all tiles
            return create_bit_indicator(stage, tile_local_offset + sort_dim_offset)
        elif stage < log2(num_tiles * NUM_SUBLANES):
            # Bit set by tile_offset, constant within tile, differs across tiles
            return create_bit_indicator(stage, tile_offset + sort_dim_offset)
        elif stage < log2(sort_dim):
            # Bit position beyond tile_offset range, tile_offset doesn't contribute
            # Pattern comes only from tile_local_offset, same for all tiles
            return create_bit_indicator(stage, sort_dim_offset + tile_local_offset)
        else:
            # Final stage(s): bit position beyond sort_dim, never set
            return create_bit_indicator(stage, sort_dim_offset)

    # Non-int stage (shouldn't happen in practice)
    return create_bit_indicator(stage, sort_dim_offset + tile_offset + tile_local_offset)


def _run_bitonic_stage_on_tiles(arrs_tiles, stage, batch_size, num_keys: int, sort_dim_offset: int = 0, sort_dim: int = None, min_stage: int = None):
    """Run a complete bitonic sort stage on tiles.

    A stage consists of multiple substages that perform comparisons at
    decreasing distances. This handles stages within compressed transpose
    format and extends to cross-lane operations for larger stages.

    Args:
        arrs_tiles: Tuple of lists of tile arrays
        stage: Current sorting stage (determines sequence length = 2^stage)
        batch_size: Batch size
        num_keys: Number of sort keys
        sort_dim_offset: Offset for bitonic order calculation
        sort_dim: Size of dimension being sorted
        min_stage: Static minimum value of stage (for dynamic stage in pl.loop)

    Returns:
        Tuple of lists of tiles with stage completed
    """
    num_tiles = len(arrs_tiles[0])
    max_substage = log2(num_tiles * NUM_SUBLANES)

    # Use min_stage for control flow when stage is dynamic (Tracer)
    stage_for_control = min_stage if type(stage) != int else stage

    if stage_for_control <= max_substage:
        # Entire stage fits within compressed transpose format
        arrs_tiles = run_compressed_transpose_format_substages_on_tiles(
            arrs_tiles,
            num_substages=stage if type(stage) == int else max_substage,
            stage=stage,
            sort_dim_offset=sort_dim_offset,
            batch_size=batch_size,
            num_keys=num_keys,
        )
    else:
        # Stage requires cross-lane operations
        # Compute tile_local_offset once (used for all cross-lane comparisons)
        # This maps lane positions to positions in the original array
        # In compressed format, lanes map to different chunks based on batch_size
        tile_local_offset = iota_tile(0) + (iota_tile(1) // batch_size) * num_tiles * NUM_SUBLANES

        # Do cross-lane substages one at a time (from high to low)
        # Each substage is independent, using the same bitonic pattern (from stage)
        for substage in range(max_substage, stage)[::-1]:
            # Do the cross-lane comparison at distance 2^substage
            # Calculate separation in lane dimension
            separation_in_lanes = 2 ** (substage - max_substage)
            lane_separation = batch_size * separation_in_lanes

            # Create permutation for cross-lane operation
            permutation = jnp.bitwise_xor(iota_tile(1), lane_separation)
            is_right_half = create_bit_indicator(log2(lane_separation), iota_tile(1))

            # Apply permutation to all tiles
            arrs_tiles_permuted = jax.tree.map(
                lambda tile: jnp.take_along_axis(tile, permutation, axis=1),
                arrs_tiles
            )

            # Compare and swap with optimized per-tile is_descending computation
            outs_tiles = [[None for _ in t] for t in arrs_tiles]
            for idx, (lefts, rights) in enumerate(zip(
                *map(transpose_list_of_lists, (arrs_tiles, arrs_tiles_permuted)),
                strict=True
            )):
                # Compute is_descending with optimizations
                # Use stage (not substage!) for the bitonic pattern - stage determines asc/desc
                is_descending_tile = _compute_is_descending_for_tile(
                    stage, idx, batch_size, num_tiles, sort_dim_offset, tile_local_offset, sort_dim
                )

                for arr_idx, out in enumerate(compare_and_swap(
                    lefts, rights,
                    is_descending=is_descending_tile,
                    is_right_half=is_right_half,
                    num_keys=num_keys
                )):
                    outs_tiles[arr_idx][idx] = out

            arrs_tiles = outs_tiles

        # After all cross-lane substages, run the remaining compressed format substages
        arrs_tiles = run_compressed_transpose_format_substages_on_tiles(
            arrs_tiles,
            num_substages=max_substage,
            stage=stage,
            sort_dim_offset=sort_dim_offset,
            batch_size=batch_size,
            num_keys=num_keys,
        )

    return arrs_tiles


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
    unpadded_sort_dim = shape[axis]

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
      num_tiles = len(arrs_tiles[0])
      sort_dim = arrs[0].shape[axis]
      num_stages = log2(sort_dim)

      # Offset to control ascending vs descending final order
      sort_dim_offset = int(descending) * sort_dim

      # Run all bitonic sort stages
      for stage in range(1, num_stages + 1):
        arrs_tiles = _run_bitonic_stage_on_tiles(
            arrs_tiles,
            stage=stage,
            batch_size=batch_size,
            num_keys=num_keys,
            sort_dim_offset=sort_dim_offset,
            sort_dim=sort_dim
        )

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
                    arrs = jax.tree.leaves([ref[...] for ref in transpose_refs])
                    arrs_tiles = [_resplit(arr, NUM_SUBLANES) for arr in arrs]

                    separation_in_lanes = 2 ** (substage_0indexed - max_substage)
                    lane_separation = batch_size * separation_in_lanes

                    permutation = jnp.bitwise_xor(iota_tile(1), lane_separation)
                    is_right_half = create_bit_indicator(log2(lane_separation), iota_tile(1))

                    arrs_tiles_permuted = jax.tree.map(
                        lambda tile: jnp.take_along_axis(tile, permutation, axis=1),
                        arrs_tiles
                    )

                    outs_tiles = [[None for _ in t] for t in arrs_tiles]
                    for idx, (lefts, rights) in enumerate(zip(
                        *map(transpose_list_of_lists, (arrs_tiles, arrs_tiles_permuted)),
                        strict=True
                    )):
                        # Compute is_descending for this tile
                        tile_offset = idx * NUM_SUBLANES
                        is_descending_tile = create_bit_indicator(
                            stage, sort_dim_offset + tile_offset + tile_local_offset
                        )

                        for arr_idx, out in enumerate(compare_and_swap(
                            lefts, rights,
                            is_descending=is_descending_tile,
                            is_right_half=is_right_half,
                            num_keys=num_keys
                        )):
                            outs_tiles[arr_idx][idx] = out

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


@functools.partial(
    jit,
    static_argnames=("num_keys", "descending", "interpret", "unroll"),
)
def bitonic_sort(
    operand: jax.Array | Sequence[jax.Array],
    num_keys: int = 1,
    descending: bool = False,
    interpret: bool = False,
    unroll: int = 128,
) -> tuple[jax.Array, ...]:
    """
    Sort arrays using bitonic sort in compressed transpose format with ref slicing.

    Optimized for sorting power-of-2 sized arrays on TPU. Uses ref slicing to
    reduce compile times by keeping data in compressed transpose format during
    sorting stages.

    Supports arbitrary input shapes - padding is handled automatically to
    nearest power of 2.

    Args:
        operand: Input array(s) of shape [batch, sort_dim].
                Can be a single array or sequence of arrays.
                Any sort_dim is supported (will be padded automatically).
        num_keys: Number of arrays to use as sort keys.
        descending: If True, sort in descending order.
        interpret: If True, run in CPU interpret mode.
        unroll: Number of tiles to allow in dim0 (default 128).

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

    # Pad to power of 2
    target_sort_dim = 2**log2(max(unpadded_shape[1], NUM_SUBLANES))
    operands = [pad(x, (NUM_SUBLANES, target_sort_dim),
      val='min' if descending else 'max') for x in operands]
    batch_size, sort_dim = operands[0].shape

    # Define output shapes
    output_shapes = [
        jax.ShapeDtypeStruct((batch_size, sort_dim), op.dtype)
        for op in operands
    ]

    # Define scratch shapes for transpose format
    # In compressed transpose format: shape is (sort_dim * batch_size / NUM_LANES, NUM_LANES)
    transpose_dim0 = (sort_dim * batch_size) // NUM_LANES
    transpose_shape = (transpose_dim0, NUM_LANES)
    scratch_shapes = [
        pltpu.VMEM(transpose_shape, to_32bit_dtype(op.dtype))
        for op in operands
    ]

    outputs = pl.pallas_call(
        functools.partial(
            bitonic_sort_refs,
            num_keys=num_keys,
            descending=descending,
            unroll=unroll,
        ),
        out_shape=(output_shapes,),
        scratch_shapes=(scratch_shapes,),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=int(0.9 * 2**27)
        ),
        interpret=interpret,
    )(operands)[0]
    return tuple(x[:unpadded_shape[0], :unpadded_shape[1]] for x in outputs)


def bitonic_topk_refs(
    in_refs,
    out_refs,
    *,
    num_keys: int,
    descending: bool,
    k: int,
    min_padded_dim0: int | None = None,
):
    """
    Pallas kernel for bitonic top-k in compressed transpose format.

    Algorithm:
    1. Pad input to satisfy alignment requirements
    2. Convert to compressed transpose format: (num_tokens, vocab) -> (128, num_tokens*chunks)
    3. Run bitonic top-k stages to select top k values per token
    4. Convert back from compressed transpose format
    5. Unpad and extract top-k per token
    """
    if not descending:
      raise NotImplementedError
    outs = bitonic_topk_arrays(
      [ref[...] for ref in in_refs], k=out_refs[0].shape[1],
      num_keys=num_keys,
      min_padded_dim0=min_padded_dim0,
    )
    for out, out_ref in zip(outs, out_refs, strict=True):
      out_ref[...] = out.astype(out_ref.dtype)


@functools.partial(
    jit,
    static_argnames=("k", "num_keys", "descending", "interpret", "min_padded_dim0"),
)
def bitonic_topk(
    operand: jax.Array | Sequence[jax.Array],
    k: int = NUM_LANES,
    num_keys: int = 1,
    descending: bool = True,
    interpret: bool = False,
    min_padded_dim0: int | None = None,
) -> tuple[jax.Array, ...]:
    """
    Compute top-k using bitonic sort in compressed transpose format.

    Optimized for k <= NUM_LANES (128). Works entirely in compressed transpose
    format for maximum TPU efficiency. Supports multiple operands like sort().

    Supports arbitrary input shapes - padding is handled automatically:
    - For small inputs (prod < NUM_LANES^2): pads dim0 to make prod = NUM_LANES^2
    - For larger inputs: pads both dims minimally to satisfy alignment

    Args:
        operand: Input array(s) of shape [num_tokens, vocab_size].
                Can be a single array or sequence of arrays.
                Any vocab_size is supported (will be padded automatically).
        k: Number of top elements (must be <= NUM_LANES=128).
        num_keys: Number of arrays to use as sort keys.
        descending: If True, sort in descending order (default for top-k).
        interpret: If True, run in CPU interpret mode.
        min_padded_dim0: Optional minimum padding for dimension 0 to tune
            performance (ALU vs. permute latency trade-off).

    Returns:
        Tuple of arrays (same length as input operands):
            - Each array has shape [num_tokens, k]

    Raises:
        ValueError: If k > NUM_LANES
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
            min_padded_dim0=min_padded_dim0,
        ),
        out_shape=(output_shapes,),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=int(0.9 * 2**27)
        ),
        interpret=interpret,
    )(operands)[0]
    return tuple(x[:unpadded_shape[0], :k] for x in outputs)
