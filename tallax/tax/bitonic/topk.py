"""
Bitonic Top-K using compressed transpose format.
"""

import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
# from jax.experimental.pallas import tpu as pltpu

from tallax.tax.utils import (
  NUM_LANES,
  NUM_SUBLANES,
  log2,
  ceil_multiple,
  pad,
  canonicalize_operand,
  to_compressed_transpose_format,
  from_compressed_transpose_format,
  to_32bit_dtype,
  join_tiles_to_array,
  split_array_to_tiles,
  map_batch_dim_to_smaller_than_hardware_tile_size,
)
from tallax.tax.bitonic.sort import (
  bitonic_sort_substage,
  _rejoin,
  _resplit,
)


def _compute_padded_shape(
  unpadded_dim0: int, unpadded_dim1: int, k: int
) -> tuple[int, int]:
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

  dim0s = [
    2**i
    for i in range(log2(NUM_SUBLANES), log2(NUM_LANES) + 1)
    if 2**i >= unpadded_dim0
  ]
  shapes = [
    (
      dim0,
      ceil_multiple(
        ceil_multiple(unpadded_dim1, NUM_LANES * NUM_LANES // dim0),
        max(k, NUM_SUBLANES),
      ),
    )
    for dim0 in dim0s
  ]
  # take minimal num elements, larger dim0 on ties as cross tile ops are faster than cross lane
  return sorted(shapes, key=lambda x: (x[0] * x[1], -x[0]))[0]


@map_batch_dim_to_smaller_than_hardware_tile_size
def bitonic_topk_arrays(
  operands: list[jax.Array],
  k: int,
  num_keys: int = 1,
  axis: int = 1,
  min_padded_dim0: int | None = None,
  presort_unroll: int | bool = True,
  merge_unroll: int | bool = True,
  transpose_refs=None,
):
  """
  Progressive bitonic merge for top-k selection.

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
      presort_unroll: Control presort (k/2 build) unrolling (int or bool)
          - True: fully unrolled (pure arrays implementation)
          - False: rolled (uses refs)
          - int: specific unroll value (m parameter for slice size)
      merge_unroll: Control merge phase unrolling (int or bool)
          - True: fully unrolled (pure arrays implementation)
          - False: rolled (uses refs)
          - int: specific unroll value (m parameter for slice size)
      transpose_refs: Scratch memory refs for rolled implementation

  Returns:
      List of JAX arrays of shape (original_batch_size, k) with top-k elements
  """
  operands, shape = canonicalize_operand(operands)
  sort_axis = axis
  batch_axis = 1 - sort_axis
  unpadded_k = k
  k = 2 ** log2(k)
  # Compute padded shape that satisfies alignment requirements
  unpadded_sort_dim = shape[sort_axis]
  if unpadded_k > unpadded_sort_dim:
    raise ValueError
  if sort_axis == 1:
    if min_padded_dim0 is None:
      min_padded_dim0 = shape[0]
    padded_shape = _compute_padded_shape(min_padded_dim0, shape[1], k=k)
  elif sort_axis == 0:
    padded_shape = (
      ceil_multiple(shape[0], max(NUM_SUBLANES, k)),
      ceil_multiple(shape[1], NUM_LANES),
    )
  else:
    raise ValueError
  # Pad both dimensions if needed
  arrs = [pad(op, block_shape=padded_shape, val="min") for op in operands]
  arrs = [x.astype(to_32bit_dtype(x.dtype)) for x in arrs]

  batch_size = arrs[0].shape[batch_axis]
  assert batch_size <= NUM_LANES
  _bitonic_sort_substage = functools.partial(
    bitonic_sort_substage, batch_size=batch_size, num_keys=num_keys
  )

  def max_reduce_stage(arrs_tiles, reduce_stage):
    for substage in range(log2(k))[::-1]:
      arrs_tiles = _bitonic_sort_substage(
        arrs_tiles, substage=substage, stage=reduce_stage
      )
    return _bitonic_sort_substage(
      arrs_tiles, substage=reduce_stage, max_reduce=True
    )

  # Convert to compressed transpose format
  if sort_axis == 1:
    arrs = jax.tree.map(to_compressed_transpose_format, arrs)
  arrs_tiles = jax.tree.map(split_array_to_tiles, arrs)
  num_tiles = len(arrs_tiles[0])
  num_merges = log2(unpadded_sort_dim) - log2(k)
  num_sublane_merges = log2(pl.cdiv(NUM_SUBLANES, k))
  num_lane_merges = log2(pl.cdiv(unpadded_sort_dim, num_tiles * NUM_SUBLANES))
  num_tile_merges = num_merges - num_sublane_merges - num_lane_merges

  # full_size is dim0 of the input in compressed transpose format
  full_size = len(arrs_tiles[0]) * arrs_tiles[0][0].shape[0]

  # Standardize unroll parameters
  if type(presort_unroll) == bool:
    presort_unroll = 1 if not presort_unroll else full_size // (2 * k)
  if type(merge_unroll) == bool:
    merge_unroll = 1 if not merge_unroll else full_size // (2 * k)

  # Compute slice sizes for presort and merge phases
  presort_slice_size = max(presort_unroll * 2 * k, NUM_SUBLANES)
  presort_slice_size = min(presort_slice_size, full_size)
  merge_slice_size = max(merge_unroll * 2 * k, NUM_SUBLANES)
  merge_slice_size = min(merge_slice_size, full_size)

  # Determine if we use rolled or unrolled implementation
  use_rolled = (presort_slice_size < full_size) or (merge_slice_size < full_size)

  if use_rolled and transpose_refs is None:
    raise ValueError("transpose_refs required when not fully unrolling")

  # Phase 1: Presort - Build bitonic sequences up to length k/2
  if presort_slice_size >= full_size:
    # Build bitonic sequences up to length k/2 (fully unrolled)
    for stage in range(1, log2(k)):
      for substage in range(stage)[::-1]:
        arrs_tiles = _bitonic_sort_substage(
          arrs_tiles, substage=substage, stage=stage
        )
  else:
    # Rolled presort using transpose_refs
    # Load data into refs
    for i, arr in enumerate(_rejoin(arrs_tiles)):
      transpose_refs[i] = transpose_refs[i].at[: arr.shape[0]]
      transpose_refs[i][...] = arr
    num_presort_slices = full_size // presort_slice_size
    remainder_presort_size = full_size % presort_slice_size

    def presort_slice(slice_i):
      slice_start = slice_i * presort_slice_size
      slice_refs = [
        ref[pl.dslice(slice_start, presort_slice_size)] for ref in transpose_refs
      ]
      # Each ref content becomes a single tile
      slice_tiles = [[ref[...]] for ref in slice_refs]

      for stage in range(1, log2(k)):
        for substage in range(stage)[::-1]:
          slice_tiles = _bitonic_sort_substage(
            slice_tiles, substage=substage, stage=stage
          )

      # Write back: tiles is list of lists, we need to concatenate inner list
      for ref, tiles_list in zip(transpose_refs, slice_tiles, strict=True):
        result = jnp.concatenate(tiles_list, axis=0)
        ref[pl.dslice(slice_start, presort_slice_size)] = result

    pl.loop(0, num_presort_slices)(presort_slice)

    # Handle remainder for presort
    if remainder_presort_size > 0:
      slice_start = num_presort_slices * presort_slice_size
      slice_refs = [
        ref[pl.dslice(slice_start, remainder_presort_size)] for ref in transpose_refs
      ]
      slice_tiles = [[ref[...]] for ref in slice_refs]

      for stage in range(1, log2(k)):
        for substage in range(stage)[::-1]:
          slice_tiles = _bitonic_sort_substage(
            slice_tiles, substage=substage, stage=stage
          )

      for ref, tiles_list in zip(transpose_refs, slice_tiles, strict=True):
        result = jnp.concatenate(tiles_list, axis=0)
        ref[pl.dslice(slice_start, remainder_presort_size)] = result

    # Convert back to arrs_tiles format only if merge phase is unrolled
    if merge_slice_size >= full_size:
      arrs_tiles = [split_array_to_tiles(ref[:full_size]) for ref in transpose_refs]

  # Phase 2: Merge - Progressive tile merges
  if merge_slice_size >= full_size:
    # Progressive merge tiles together as far as possible first (fully unrolled)
    for _ in range(num_tile_merges):
      # special handling for cross tile as tile to compare to may not exist
      remainder_length = len(arrs_tiles[0]) % (2 * pl.cdiv(k, NUM_SUBLANES))
      if remainder_length:
        remainder_arrs_tiles = [x[-remainder_length:] for x in arrs_tiles]
        arrs_tiles = [x[:-remainder_length] for x in arrs_tiles]
      arrs_tiles = max_reduce_stage(
        arrs_tiles, reduce_stage=log2(ceil_multiple(k, NUM_SUBLANES))
      )
      if remainder_length:
        arrs_tiles = [
          x + rem for x, rem in zip(arrs_tiles, remainder_arrs_tiles, strict=True)
        ]
  else:
    # Rolled merge - load into refs if not already there
    if presort_slice_size >= full_size:
      # Data is still in arrs_tiles, need to load into refs
      for i, arr in enumerate(_rejoin(arrs_tiles)):
        transpose_refs[i] = transpose_refs[i].at[: arr.shape[0]]
        transpose_refs[i][...] = arr

    # Rolled merge with decreasing active size
    # Flattened loop over both num_tile_merges and inner slices
    pair_size = pl.cdiv(k, NUM_SUBLANES)
    active_size = full_size

    for merge_iter in range(num_tile_merges):
      # Check for remainder that doesn't have a pair
      remainder_length = active_size % (2 * pair_size)

      # Number of pairs to process (each pair is 2*pair_size elements)
      num_pairs = active_size // (2 * pair_size)

      # Process pairs in chunks of merge_unroll
      num_slice_iterations = (num_pairs + merge_unroll - 1) // merge_unroll

      def process_merge_slices(slice_iter):
        # Process merge_unroll pairs at a time
        start_pair = slice_iter * merge_unroll
        end_pair = jnp.minimum((slice_iter + 1) * merge_unroll, num_pairs)
        num_pairs_in_slice = end_pair - start_pair

        # Process each pair in this slice
        for local_pair_i in range(merge_unroll):
          @pl.when(local_pair_i < num_pairs_in_slice)
          def process_pair():
            pair_i = start_pair + local_pair_i
            slice_start = pair_i * 2 * pair_size
            slice_size_for_pair = 2 * pair_size

            # Read the slice
            slice_arrs = [
              ref[pl.dslice(slice_start, slice_size_for_pair)] for ref in transpose_refs
            ]

            # Split into tiles for max_reduce_stage
            slice_tiles = [split_array_to_tiles(arr) for arr in slice_arrs]

            # Apply max_reduce_stage
            reduced_tiles = max_reduce_stage(
              slice_tiles, reduce_stage=log2(ceil_multiple(k, NUM_SUBLANES))
            )

            # Rejoin tiles and write back only the top half
            for ref, tiles_list in zip(transpose_refs, reduced_tiles, strict=True):
              result = jnp.concatenate(tiles_list, axis=0)
              ref[pl.dslice(slice_start, result.shape[0])] = result

      if num_slice_iterations > 0:
        pl.loop(0, num_slice_iterations)(process_merge_slices)

      # After processing all pairs, move remainder to follow reduced pairs
      # New active size = reduced pairs + remainder
      new_active_size = num_pairs * pair_size + remainder_length
      if remainder_length > 0:
        # Move remainder from [num_pairs * 2 * pair_size : num_pairs * 2 * pair_size + remainder_length]
        # to [num_pairs * pair_size : num_pairs * pair_size + remainder_length]
        remainder_src_start = num_pairs * 2 * pair_size
        remainder_dst_start = num_pairs * pair_size
        for ref in transpose_refs:
          ref[pl.dslice(remainder_dst_start, remainder_length)] = ref[pl.dslice(remainder_src_start, remainder_length)]

      active_size = new_active_size

    # Back in array flow - extract final reduced size and split into tiles
    arrs_tiles = [split_array_to_tiles(ref[:active_size]) for ref in transpose_refs]

  for i in range(num_lane_merges)[::-1]:
    arrs_tiles = max_reduce_stage(
      arrs_tiles, reduce_stage=log2(ceil_multiple(k, NUM_SUBLANES)) + i
    )
  for i in range(num_sublane_merges)[::-1]:
    arrs_tiles = max_reduce_stage(arrs_tiles, reduce_stage=log2(k) + i)

  # Final sort: convert bitonic sequence to fully descending order
  # Use sort_dim_offset=k to ensure descending direction
  for substage in range(log2(k))[::-1]:
    arrs_tiles = _bitonic_sort_substage(
      arrs_tiles, substage=substage, stage=log2(k), sort_dim_offset=k
    )

  if sort_axis == 1:
    arrs = [
      from_compressed_transpose_format(tiles, dim0=batch_size)
      for tiles in arrs_tiles
    ]
    return [arr[: shape[batch_axis], :unpadded_k] for arr in arrs]
  else:
    arrs = [
      join_tiles_to_array(tiles, dim0=ceil_multiple(k, NUM_SUBLANES))
      for tiles in arrs_tiles
    ]
    return [arr[:unpadded_k, : shape[batch_axis]] for arr in arrs]


def max_arrays(operands, num_keys, axis):
  arrs = bitonic_topk_arrays(operands, num_keys=num_keys, k=1, axis=axis)
  return [x.squeeze(axis) for x in arrs]
