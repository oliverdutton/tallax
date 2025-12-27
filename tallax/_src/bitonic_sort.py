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


Before
this is better than before when substages were all compiled (num_stages - substage) times. so for 2**15=32768 thats 15,14,13,12,11,10,9,8,7,6,5,4,3,2,1 substages = 120 times
- 42+6=48 permute compilations
- 72 tile compilations

After
This version does (num sublane stages, num tile stages, num lane stages)
there are 1+2+3=6 sublane compilations in the stage unrolled first 3 stages
the tile substages get compiled 2 times
the lane substages get compiled once

so for (16, 32768). 6 sublane, 9 tile, 3 lane.
- 9 permute compilations
- 18 tile compilations

Is the permute compilation dominating? - add skip_permutes kwarg and check T/F

the issue is still linear compile and trace times due to number of tiles


"""
import functools
from functools import lru_cache
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
from tallax._src.symint import SymInt, unwrap


def set_cummax(vs):
  o = [vs[0]]
  for v in vs[1:]:
    if v > o[-1]:
      o.append(v)
  return type(vs)(o)


def compare_and_swap(lefts, rights, *, num_keys: int, is_descending: jax.Array | None, is_right_half=None,
             has_unique_key=False):
  """Compare and conditionally swap array pairs.

  Args:
    lefts: Tuple of left arrays to compare
    rights: Tuple of right arrays to compare
    num_keys: Number of arrays to use as sort keys
    is_descending: Boolean mask for sort direction (None implies ascending)
    is_right_half: Mask for subtile comparisons. Needed for handling ties in values correctly.
    has_unique_key: Whether first key is guaranteed unique (optimizes sort)

  Returns:
    Tuple of (sorted_lefts, sorted_rights) or sorted values for subtile.
  """
  num_arrs = len(lefts)

  def _compare_pair(i, left, right):
    handle_subtile_ties = (
        is_right_half is not None
        and not has_unique_key and num_arrs != num_keys and i == num_keys - 1
    )

    if handle_subtile_ties:
      left, right = (
          jnp.where(is_right_half, right, left),
          jnp.where(is_right_half, left, right)
      )

    mask = (left > right if type(is_descending) == bool and is_descending
            else right > left)
    mask = mask.astype(jnp.int32)

    if is_right_half is not None and not handle_subtile_ties:
      mask = jnp.bitwise_xor(mask, is_right_half.astype(jnp.int32))
    return mask

  masks = tuple(
      _compare_pair(i, left, right)
      for i, (left, right) in enumerate(zip(lefts, rights, strict=True))
  )

  ties = [(left == right) for left, right in zip(lefts, rights, strict=True)]

  mask = masks[0]
  for k in range(1, num_keys):
    # Break ties in primary key with secondary key comparison
    mask = jnp.where(ties[k - 1], masks[k], mask)
    ties[k] &= ties[k - 1]

  if is_descending is not None and type(is_descending) != bool:
    # Dynamic descending mask
    mask = mask.astype(bool)
    is_descending = is_descending.astype(bool)
    mask = mask ^ is_descending

  return jax.tree.map(
      lambda left, right: (
          (jnp.where(mask, left, right), jnp.where(mask, right, left))
          if is_right_half is None else
          jnp.where(mask, left, right)
      ),
      lefts, rights
  )


@lru_cache
def compute_pair_slice_start_index(i, separation, slice_length=1):
    """Compute start index for pair-wise array slicing."""
    if slice_length > separation:
      raise ValueError(
          f'Separation must be at least slice length, {separation=} {slice_length=}'
      )
    slices_per_pair = separation // slice_length
    pair_idx = i // slices_per_pair
    slice_idx = i % slices_per_pair
    return pair_idx * 2 * separation + slice_idx * slice_length


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
    dim1 = 2**log2(ceil_multiple(unpadded_dim1, max(k, NUM_SUBLANES)))
    return (dim0, dim1)

  dim0s = [2**i for i in range(log2(NUM_SUBLANES), log2(NUM_LANES)+1)
    if 2**i >= unpadded_dim0]
  shapes = [
    (dim0, 2**log2(ceil_multiple(unpadded_dim1,
      NUM_LANES * NUM_LANES // dim0)))
    for dim0 in dim0s]
  # take minimal num elements, larger dim0 on ties as cross tile ops are faster than cross lane
  return sorted(shapes, key=lambda x: (x[0] * x[1], -x[0]))[0]


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


def _rejoin(operands):
  def _inner(operand):
    tiles = jax.tree.leaves(operand)
    return jnp.concatenate(tiles, axis=0)
  return [_inner(x) for x in operands]


def reverse_tiles(arr):
  return jnp.concatenate(
    jnp.split(arr, arr.shape[0]//NUM_SUBLANES, 0)[::-1],
    axis=0)


def concrete_and_true(b):
  return (type(b)==bool and b)


def _compute_is_descending(stage: SymInt | int, tile_start_offset: SymInt | int, tile_local_offset: jax.Array, sort_dim_offset: SymInt | int, compression_length: int, substage: int | None=None):
    # is_descending repeats every 2**(stage+1)
    # Optimize sort_dim_offset if
    if concrete_and_true(
        (sort_dim_offset % (2**(stage+1))) < 2**stage
    ):
      sort_dim_offset = 0
    if concrete_and_true(
        (sort_dim_offset % (2**(stage+1))) >= 2**stage
    ):
      sort_dim_offset = 2**stage


    # Check if we can optimize based on stage comparisons
    if concrete_and_true(stage < log2(NUM_SUBLANES)) or concrete_and_true(stage >= log2(compression_length)):
      # Same pattern for all tiles
      return create_bit_indicator(unwrap(stage), tile_local_offset + unwrap(sort_dim_offset))

    if concrete_and_true(stage >= log2(NUM_SUBLANES)) and concrete_and_true(stage < log2(compression_length)):
        # Bit set by tile_offset, constant within tile, differs across tiles
        return create_bit_indicator(unwrap(stage), tile_start_offset + unwrap(sort_dim_offset))

    # Can't optimize - use full computation
    return create_bit_indicator(unwrap(stage), tile_start_offset + tile_local_offset + unwrap(sort_dim_offset))


def bitonic_sort_substage(arrs_tiles, *, substage, num_keys: int, batch_size: int, stage: SymInt | int | None = None, sort_dim_offset: int = 0, compression_length:int=None, concat_threshold: int | None = None, max_reduce: bool = False):
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
    assert max_reduce or stage is not None
    separation = 2**substage
    # if still arrays, we make it into one big tile so its sanitized to list[list[jax.ndarray]]
    arrs_tiles = list(map(jax.tree.leaves, arrs_tiles))
    if compression_length is None:
      compression_length = len(arrs_tiles[0]) * arrs_tiles[0][0].shape[0]
    if separation < NUM_SUBLANES or separation >= compression_length:
      # we need to permute within tiles
      axis = int(separation >= compression_length)
      intra_tile_separation = separation if axis==0 else ((separation * batch_size) // compression_length)

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
              compression_length=compression_length,
              substage=substage,
            ) if not max_reduce else True,
            is_right_half=is_right_half,
            num_keys=num_keys
          )):
            outs_tiles[arr_idx][idx] = out
    else:
      # Comparison between tiles

      # concatting tiles simplifies the code, but hides optimizationsy from the compiler. So until tiles are large (the concat_threshold) we keep them as hardware tile size
      tile_size = separation if ((concat_threshold is not None) and (separation >= concat_threshold)) else NUM_SUBLANES

      arrs_tiles = _resplit(arrs_tiles, tile_size)
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
              compression_length=compression_length,
              substage=substage,
            ) if not max_reduce else True,
            num_keys=num_keys
        )):
          outs_tiles[arr_idx][idx] = out_left
          if not max_reduce:
            outs_tiles[arr_idx][idx + tile_separation] = out_right
    if max_reduce:
      # remove the Nones, the lower half we discard for top-k usage
      outs_tiles = [[v for v in out_tiles if v is not None] for out_tiles in outs_tiles]
    assert all(not any([v is None for v in out_tiles]) for out_tiles in outs_tiles)
    return outs_tiles


def _bitonic_sort_arrays(arrs_tiles, stage_unroll, num_stages, sort_dim_offset, slice_size, num_keys, batch_size, compression_length):
  
  sort_kwargs = dict(num_keys=num_keys, batch_size=batch_size, compression_length=compression_length)
  out_arrs_tiles = []
  for i, arrs_slice_tiles in enumerate(transpose_list_of_lists(_resplit(arrs_tiles, slice_size))):
    for stage in range(1, stage_unroll + 1):
      for substage in range(stage)[::-1]:
        arrs_slice_tiles = bitonic_sort_substage(arrs_slice_tiles, substage=substage, stage=stage, **sort_kwargs, sort_dim_offset=(sort_dim_offset+i*slice_size) % (2**(stage+1)), )
    out_arrs_tiles.append([jnp.concat(x, axis=0) for x in arrs_slice_tiles])
  arrs_tiles = transpose_list_of_lists(out_arrs_tiles)

  for stage in range(stage_unroll + 1, num_stages + 1):
    for substage in range(stage_unroll, stage)[::-1]:
      arrs_tiles = bitonic_sort_substage(arrs_tiles, substage=substage, stage=stage, sort_dim_offset=sort_dim_offset,
      concat_threshold=slice_size,
      **sort_kwargs,
      )

    out_arrs_tiles = []
    for i, arrs_slice_tiles in enumerate(transpose_list_of_lists(_resplit(arrs_tiles, slice_size))):
      for substage in range(stage_unroll)[::-1]:
        arrs_slice_tiles = bitonic_sort_substage(arrs_slice_tiles, substage=substage, stage=stage, sort_dim_offset=(sort_dim_offset+i*slice_size) % (2**(stage+1)), **sort_kwargs)
      out_arrs_tiles.append([jnp.concat(x, axis=0) for x in arrs_slice_tiles])
    arrs_tiles = transpose_list_of_lists(out_arrs_tiles)
  return arrs_tiles


def _bitonic_sort_substages_refs(transpose_refs, *, substages, stages, num_keys: int, batch_size: int, sort_dim_offset: int = 0, compression_length=None, slice_size=None, ref_slice_size=None, concat_threshold=None):
  if ref_slice_size is None:
    ref_slice_size = compression_length
  if slice_size is None:
    slice_size = ref_slice_size
  if concat_threshold is None:
    concat_threshold = slice_size
  slice_size = min(slice_size, ref_slice_size)

  # checks if the sharding of input is compatible with the substage comparison separation, splitting it up into subsections if not
  sharded = tuple(2**substage < slice_size for substage in substages)
  if all(sharded):
    pass
  elif all((not b for b in sharded)):
    ref_slice_size = compression_length
    slice_size = compression_length
  else:
    # will switch between running on ref slices and whole ref. We do the longest run we can of same slice_size to minimize ref read/writes
    split_i = next(i for i, v in enumerate(sharded) if v!=sharded[0])
    [_bitonic_sort_substage_refs(
        transpose_refs,
        substages=substages, stages=stages,
        num_keys=num_keys, batch_size=batch_size,
        sort_dim_offset=sort_dim_offset, compression_length=compression_length,
        ref_slice_size=ref_slice_size,
        slice_size=slice_size,
        concat_threshold=concat_threshold,
    ) for substages, stages in [
        (substages[:split_i], stages[:split_i]),
        (substages[split_i:], stages[split_i:])]]
    return

  #print(f'{any(sharded)=} {substages=} {slice_size=} {ref_slice_size=} {stages=}')
  grid_size = compression_length // ref_slice_size

  def process_block(i):
    arrs_tiles = [
        ref[pl.dslice(i * ref_slice_size, ref_slice_size)]
        for ref in transpose_refs]
    out_arrs_tiles = []
    for j, arrs_slice_tiles in enumerate(transpose_list_of_lists(_resplit(arrs_tiles, slice_size))):
      tile_offset = sort_dim_offset + SymInt(i, 0, grid_size-1) * ref_slice_size + SymInt(j) * slice_size
      for substage, stage in zip(substages, stages, strict=True):
        arrs_slice_tiles = bitonic_sort_substage(
            arrs_slice_tiles,
            substage=substage,
            stage=stage,
            num_keys=num_keys,
            batch_size=batch_size,
            sort_dim_offset=tile_offset,
            compression_length=compression_length,
            concat_threshold=concat_threshold)
      out_arrs_tiles.append([jnp.concat(x, axis=0) for x in arrs_slice_tiles])
    arrs_tiles = transpose_list_of_lists(out_arrs_tiles)

    # Write back to refs
    for ref, arr in zip(transpose_refs, _rejoin(arrs_tiles), strict=True):
      ref[pl.dslice(i * ref_slice_size, ref_slice_size)] = arr

  if grid_size == 1:
    process_block(0)
  else:
    pl.loop(0, grid_size)(process_block)


def bitonic_sort_maybe_rolled(operands: list[jax.Array], num_keys: int = 1, axis: int = 1, descending: bool = False, stage_unroll: int | None = None, slice_size_unroll: int | None = None, unroll_stages: bool = True, ref_slice_size_unroll: int | None = None, transpose_refs=None, num_stages: int | None = None, single_stage: jax.Array | None = None, sort_dim_offset: SymInt | int | None = None):
    """
    Bitonic sort using compressed transpose format, , offers both rolled and
    fully unrolled implementation.

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
    if stage_unroll is not None:
      assert transpose_refs is not None
    
    if single_stage is not None:
      # special code path for large inputs which dont fit in VMEM
      assert not unroll_stages

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
    # we put it at the neccessary side to avoid padding leakage in stable sorts
    arrs = [pad(op, block_shape=padded_shape, val='min' if descending else 'max', prepend=(False, descending)) for op in operands]
    arrs = [x.astype(to_32bit_dtype(x.dtype)) for x in arrs]

    num_stages = log2(shape[axis]) if num_stages is None else num_stages
    stage_unroll = min(stage_unroll, num_stages) if stage_unroll is not None else num_stages
    # Offset to control ascending vs descending final order
    sort_dim_offset = int(descending) * (2**num_stages) if sort_dim_offset is None else sort_dim_offset

    def _sort_arrays(arrs):
      batch_size = arrs[0].shape[batch_axis]
      assert batch_size <= NUM_LANES
      # Convert to compressed transpose format
      arrs_tiles = jax.tree.map((to_compressed_transpose_format if axis==1 else split_array_to_tiles), arrs)
      compression_length = arrs_tiles[0].shape[0]

      sort_kwargs = dict(num_keys=num_keys, batch_size=batch_size, compression_length=compression_length,)

      slice_size = 2**stage_unroll if unroll_stages else NUM_SUBLANES
      if slice_size_unroll is not None:
        slice_size = max(slice_size, 2**slice_size_unroll)
      ref_slice_size = compression_length
      if ref_slice_size_unroll is not None:
        ref_slice_size = min(
          max(slice_size, 2**ref_slice_size_unroll),
          ref_slice_size)
      # clip the slice size
      slice_size, ref_slice_size = (min(max(size, NUM_SUBLANES), compression_length) for size in (slice_size, ref_slice_size))

      if unroll_stages:
        arrs_tiles = _bitonic_sort_arrays(arrs_tiles, stage_unroll, num_stages, sort_dim_offset, slice_size, **kwargs)
      else:
        # use the transpose refs
        for ref, arr in zip(transpose_refs, _rejoin(arrs_tiles), strict=True):
          ref[...] = arr

        num_crosslane_stages = log2(NUM_LANES // batch_size)
        stage_sections = set_cummax((
          stage_unroll,
          # two sections added to allow for is_descending optimization
          # specializing for constant intra-tile from constant across tiles patterns
          num_stages - num_crosslane_stages - 1,
          num_stages,
        ))
        stage_sections = tuple(i+1 for i in stage_sections) # stages are 1-indexed

        stages, substages = [],[]
        for stage in range(1, stage_sections[0]):
          for substage in range(stage)[::-1]:
            stages.append(stage)
            substages.append(substage)
        
        # special code branch for sorting things which dont fit in HBM
        if single_stage is not None:
          substages = tuple(range(num_stages)[::-1])
          stages = (single_stage,)*num_stages
          stage_sections = (0,)
  
        _bitonic_sort_substages_refs(
          transpose_refs, substages=substages, stages=stages, **sort_kwargs, sort_dim_offset=sort_dim_offset, slice_size=slice_size,
          ref_slice_size=ref_slice_size,
        )

        for stage_lb, stage_ub in zip(stage_sections, stage_sections[1:]):
          # run the cross tile and cross lane fori_loops separately so we can make optimizations on is_descending
          @pl.loop(stage_lb, stage_ub)
          def run_dynamic_stage(stage):
            # bounds are inclusive on both ends
            # this is used to make optimizations on is_descending inside the code
            stage = SymInt(stage, lower_bound=stage_lb, upper_bound=stage_ub-1)

            for substage in range(stage_lb, stage_ub)[::-1]:
              @pl.when(stage > substage)
              def run_substage():
                _bitonic_sort_substages_refs(
                  transpose_refs, substages=(substage,), stages=(stage,), sort_dim_offset=sort_dim_offset, **sort_kwargs, slice_size=slice_size, ref_slice_size=ref_slice_size,
                )

            substages = tuple(range(stage_lb)[::-1])
            stages = (stage,)*len(substages)
            _bitonic_sort_substages_refs(
            transpose_refs, substages=substages, stages=stages, sort_dim_offset=sort_dim_offset, **sort_kwargs, slice_size=slice_size, ref_slice_size=ref_slice_size)
        # back in array flow
        arrs_tiles = [[ref[...]] for ref in transpose_refs]

      # Convert back from compressed transpose format
      if axis == 1:
        arrs = [from_compressed_transpose_format(tiles, dim0=batch_size) for tiles in arrs_tiles]
      else:
        arrs = [join_tiles_to_array(tiles, dim0=ceil_multiple(2**num_stages, NUM_SUBLANES)) for tiles in arrs_tiles]
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
    return [(arr[:shape[0], :shape[1]] if descending else arr[:shape[0], -shape[1]:]) for arr in arrs]

