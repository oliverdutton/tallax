
import functools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax._src.utils import (
    is_cpu_platform,
    log2,
    max_int,
    all_concrete_ints,
    pad,
    float_to_sortable_int,
    sortable_int_to_float,
    pack_bf16_u16_to_i32,
    unpack_bf16_u16_from_i32,
    split_array_to_tiles,
    join_tiles_to_array,
    iota_tile,
    create_bit_indicator,
    convert_to_sublane_sort_format,
    convert_from_sublane_sort_format,
    transpose_list_of_lists,
    to_32bit_dtype,
    same_shape_dtype,
    canonicalize_operand,
    is_32bit,
    NUM_LANES,
    NUM_SUBLANES,
)


### Bitonic Sort Core Operations

def compare_and_swap(lefts, rights, num_keys: int, is_descending: jax.Array | None, is_right_half=None,
             has_unique_key=False):
  """Compare and conditionally swap array pairs.

  Args:
    lefts: Tuple of left arrays to compare
    rights: Tuple of right arrays to compare
    num_keys: Number of arrays to use as sort keys
    is_descending: Boolean mask for sort direction
    is_right_half: Mask for subtile comparisons
    has_unique_key: Whether first key is guaranteed unique

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


### Cross-Tile Substage

def _crosstile_substage_refs(
    refs,
    substage: int,
    stage: int,
    num_keys: int,
    unroll: int = 16,
    dim1_offset: int = 0,
):
  """Pallas kernel for cross-tile substage (large stages).

  Args:
    refs: References to arrays being sorted
    substage: Current substage within stage
    stage: Current sorting stage
    num_keys: Number of arrays to use as keys
    unroll: Loop unrolling factor
    dim1_offset: Offset for bitonic order calculation
  """
  assert (unroll % 2) == 0, 'Static sort order requires even unroll factor'

  num_pairs = refs[0].shape[-1] // 2 ** (substage + 1)
  unroll = min(unroll, num_pairs)

  @pl.loop(0, pl.cdiv(num_pairs, unroll))
  def process_pairs(loop_idx):
    pair_length = 2 ** (substage + 1)
    slice_length = unroll * pair_length
    ref_slices = [
        ref.at[:, pl.dslice(loop_idx * slice_length, slice_length)]
        for ref in refs
    ]

    outs = [[] for _ in refs]
    for i in range(unroll):
      pair_offset = (loop_idx * unroll + i) * pair_length
      half_length = 2 ** substage

      lefts = [v[:, i * pair_length: i * pair_length + half_length]
               for v in ref_slices]
      rights = [v[:, i * pair_length + half_length:
                   i * pair_length + 2 * half_length]
                for v in ref_slices]

      is_descending = create_bit_indicator(stage, dim1_offset + pair_offset)

      for i, vs in enumerate(compare_and_swap(lefts, rights,
                                      is_descending=is_descending,
                                      num_keys=num_keys)):
        outs[i].extend(vs)

    for ref_slice, out in zip(ref_slices, outs, strict=True):
      ref_slice[...] = jnp.concatenate(out, axis=-1)


### Within-Tile Substages

def _compute_pair_slice_start_index(i, separation, slice_length=1):
  """Compute start index for pair-wise array slicing."""
  if slice_length > separation:
    raise ValueError(
        f'Separation must be at least slice length, {separation=} {slice_length=}'
    )
  slices_per_pair = separation // slice_length
  pair_idx = i // slices_per_pair
  slice_idx = i % slices_per_pair
  return pair_idx * 2 * separation + slice_idx * slice_length


def _packed_subtile_substage(substage, arrs_tiles, *, stage, dim1_offset,
                              num_keys: int, dim0: int):
  """Single substage using sublane permutation (packed, within tiles)."""
  assert dim0 is not None
  index = iota_tile(0)
  global_base_index = iota_tile(0) + (((iota_tile(1) // dim0) * NUM_LANES))
  tile_rows = NUM_LANES // NUM_SUBLANES
  tile_cols = len(arrs_tiles[0]) // tile_rows

  is_right_half = create_bit_indicator(substage, index)
  permutation = jnp.bitwise_xor(index, 1 << substage)

  arrs_tiles_permuted = jax.tree.map(
      lambda tile: jnp.take_along_axis(tile, permutation, axis=0),
      arrs_tiles
  )

  outs_tiles = [[] for _ in arrs_tiles]

  for tile_idx, (lefts, rights) in enumerate(zip(
      *map(transpose_list_of_lists, (arrs_tiles, arrs_tiles_permuted)),
      strict=True
  )):
    tile_row = tile_idx // tile_cols
    tile_col = tile_idx % tile_cols
    tile_offset = (tile_row * NUM_SUBLANES +
                   tile_col * (NUM_LANES * (NUM_LANES // dim0)))

    is_descending = create_bit_indicator(
        stage, dim1_offset + tile_offset + global_base_index
    )

    if type(stage) == int:
      # Performance optimizations for early, statically compiled stages
      if stage < log2(NUM_SUBLANES):
        is_descending = create_bit_indicator(stage, global_base_index)
      elif stage < log2(NUM_LANES):
        is_descending = create_bit_indicator(stage, tile_offset)

    for i, o in enumerate(compare_and_swap(lefts, rights,
                                   is_descending=is_descending,
                                   is_right_half=is_right_half,
                                   num_keys=num_keys)):
      outs_tiles[i].append(o)

  return outs_tiles


def _packed_crosstile_substage(substage, arrs_tiles, *, stage, dim1_offset,
                                num_keys: int, dim0: int):
  """Single substage using lane permutation (packed, between tiles)."""
  index = global_base_index = iota_tile(1)
  tile_rows = dim0 // NUM_SUBLANES
  tile_cols = len(arrs_tiles[0]) // tile_rows

  is_right_half = create_bit_indicator(substage, index)
  permutation = jnp.bitwise_xor(index, 1 << substage)

  arrs_tiles_permuted = jax.tree.map(
      lambda tile: jnp.take_along_axis(tile, permutation, axis=1),
      arrs_tiles
  )

  outs_tiles = [[] for _ in arrs_tiles]

  for tile_idx, (lefts, rights) in enumerate(zip(
      *map(transpose_list_of_lists, (arrs_tiles, arrs_tiles_permuted)),
      strict=True
  )):
    tile_offset = (tile_idx % tile_cols) * NUM_LANES

    is_descending = create_bit_indicator(
        stage, dim1_offset + tile_offset + global_base_index
    )

    if type(stage) == int:
      # Performance optimizations for early, statically compiled stages
      if stage < log2(NUM_SUBLANES):
        is_descending = create_bit_indicator(stage, global_base_index)
      elif stage < log2(NUM_LANES):
        is_descending = create_bit_indicator(stage, tile_offset)

    for i, o in enumerate(compare_and_swap(lefts, rights,
                                   is_descending=is_descending,
                                   is_right_half=is_right_half,
                                   num_keys=num_keys)):
      outs_tiles[i].append(o)

  return outs_tiles


def _crosstile_substage(arrs_tiles, substage, dim0, num_keys: int,
                        dim1_offset=0, stage=None):
  """Single substage using explicit tile pair comparison."""
  global_base_index = iota_tile(0) + (((iota_tile(1) // dim0) * NUM_LANES))
  num_tiles = len(arrs_tiles[0])
  tile_rows = NUM_LANES // NUM_SUBLANES
  tile_cols = num_tiles // tile_rows

  separation = (2**substage // NUM_SUBLANES) * tile_cols
  outs_tiles = [[None for _ in t] for t in arrs_tiles]

  for i in range(num_tiles // 2):
    idx = _compute_pair_slice_start_index(i, separation=separation)

    tile_row = idx // tile_cols
    tile_col = idx % tile_cols
    pair_offset = (tile_row * NUM_SUBLANES +
                   tile_col * (NUM_LANES * (NUM_LANES // dim0)))
    lefts, rights = (
        transpose_list_of_lists(arrs_tiles)[j]
        for j in (idx, idx + separation)
    )

    is_descending = create_bit_indicator(
        stage, dim1_offset + pair_offset + global_base_index
    )

    if type(stage) == int and stage < log2(NUM_LANES):
      is_descending = create_bit_indicator(stage, pair_offset)

    for i, (o_left, o_right) in enumerate(compare_and_swap(
        lefts, rights, is_descending=is_descending, num_keys=num_keys
    )):
      outs_tiles[i][idx] = o_left
      outs_tiles[i][idx + separation] = o_right

  assert all(
      not any([v is None for v in out_tiles])
      for out_tiles in outs_tiles
  )
  return outs_tiles


def _packed_substages_arrays(
    arrs_tiles,
    num_substages: int,
    stage: int,
    dim0: int,
    num_keys: int,
    dim1_offset: int = 0,
):
  """Execute multiple substages using packed methods.

  Dispatches to:
  - _packed_subtile_substage for substage < log2(NUM_SUBLANES)
  - _packed_crosstile_substage for substage >= log2(NUM_SUBLANES)
  """
  assert num_substages <= log2(NUM_LANES)

  for substage in range(num_substages)[::-1]:
    if substage >= log2(NUM_SUBLANES):
      # Inter-tile comparisons using lane permute
      arrs_tiles = _packed_crosstile_substage(
          substage, arrs_tiles, stage=stage,
          dim0=dim0, dim1_offset=dim1_offset, num_keys=num_keys
      )
    else:
      # Intra-tile comparisons using sublane permute
      arrs_tiles = _packed_subtile_substage(
          substage, arrs_tiles, stage=stage,
          dim0=dim0, dim1_offset=dim1_offset, num_keys=num_keys
      )
  return arrs_tiles


def _packed_substages_refs(
    refs,
    *,
    num_substages: int,
    stage: int,
    num_keys: int,
    unroll: int = 256,
    dim1_offset: int = 0,
    slice_dim1: int = None,
):
  """Pallas kernel wrapper for _packed_substages_arrays."""
  shape = refs[0].shape
  if slice_dim1 is None:
    slice_dim1 = min(unroll * NUM_LANES, shape[1])

  unroll_dim0 = (unroll * NUM_LANES) // slice_dim1
  slice_dim0 = min(unroll_dim0 * NUM_SUBLANES, shape[0])
  unroll = (slice_dim0 * slice_dim1) // (NUM_SUBLANES * NUM_LANES)

  grid_dim0 = shape[0] // slice_dim0
  grid_dim1 = shape[1] // slice_dim1

  @pl.loop(0, grid_dim0 * grid_dim1)
  def process_block(loop_idx):
    block_row = loop_idx // grid_dim1
    block_col = loop_idx % grid_dim1

    ref_slices = [
        ref.at[
            pl.dslice(block_row * slice_dim0, slice_dim0),
            pl.dslice(block_col * slice_dim1, slice_dim1)
        ]
        for ref in refs
    ]

    slice_shape = ref_slices[0].shape
    
    # pad in dim0 (if needed)
    arrs = [pad(ref_slice[...], block_shape=(
        pl.cdiv(NUM_LANES * NUM_LANES, slice_shape[1]), slice_shape[1]))
        for ref_slice in ref_slices]
    dim0 = arrs[0].shape[0]
    arrs_tiles = jax.tree.map(convert_to_sublane_sort_format, arrs)

    if stage is not None:
      # Run single stage
      arrs_tiles = _packed_substages_arrays(
          arrs_tiles,
          num_substages=num_substages,
          stage=stage,
          dim1_offset=dim1_offset + (block_col * slice_dim1),
          dim0=dim0,
          num_keys=num_keys,
      )
    else:
      # Run all stages 1 to num_substages (allows compiler fusion)
      num_stages = num_substages
      for stage_ in range(1, num_stages + 1):
        arrs_tiles = _packed_substages_arrays(
            arrs_tiles,
            num_substages=stage_,
            stage=stage_,
            dim1_offset=dim1_offset + (block_col * slice_dim1),
            dim0=dim0,
            num_keys=num_keys,
        )

    outs = [
        convert_from_sublane_sort_format(tiles, dim0=dim0)[:slice_shape[0]]
        for tiles in arrs_tiles
    ]

    for ref_slice, out in zip(ref_slices, outs, strict=True):
      ref_slice[...] = out


### Stage Execution

def _stages(
    start_stage: int,
    end_stage: int,
    refs,
    num_keys: int,
    unroll_crosstile: int = 64,
    unroll_subtile: int = 64,
    dim1_offset: int = 0,
    start_stage_static_lower_bound: int | None = None
):
  """Execute range of bitonic sorting stages."""
  log_n = log2(refs[0].shape[1])

  if start_stage_static_lower_bound is None:
    start_stage_static_lower_bound = start_stage

  # Run stages 1 to 7 (if large enough), compiler fused
  if start_stage_static_lower_bound == 1:
    _packed_substages_refs(
        refs,
        num_substages=min(log2(NUM_LANES), end_stage),
        stage=None,
        dim1_offset=dim1_offset,
        unroll=unroll_subtile,
        num_keys=num_keys,
    )
  elif (all_concrete_ints(start_stage, end_stage)
        and start_stage <= log2(NUM_LANES) and end_stage == start_stage + 1):
    _packed_substages_refs(
        refs,
        num_substages=start_stage,
        stage=start_stage,
        dim1_offset=dim1_offset,
        unroll=unroll_subtile,
        num_keys=num_keys,
    )
    return
  else:
    assert start_stage_static_lower_bound > log2(NUM_LANES), \
        'stages 1 to log2(NUM_LANES) only triggered as fully unrolled code block'

  # Run stages 8 and upwards
  @pl.loop(max_int(start_stage, log2(NUM_LANES) + 1), end_stage)
  def run_stage(stage):
    for substage in range(log2(NUM_LANES), log_n)[::-1]:
      # Run substages 7 and up
      @pl.when(stage > substage)
      def _():
        _crosstile_substage_refs(
            refs,
            substage=substage,
            stage=stage,
            unroll=unroll_crosstile,
            dim1_offset=dim1_offset,
            num_keys=num_keys,
        )

    # Run substages 0-6 inclusive
    _packed_substages_refs(
        refs,
        num_substages=log2(NUM_LANES),
        stage=stage,
        dim1_offset=dim1_offset,
        unroll=unroll_subtile,
        num_keys=num_keys,
    )


### Main Bitonic Sort Kernel

def bitonic_sort(
    refs,
    stage_ref,
    *,
    num_keys: int,
    descending: bool | None = None,
    log_n: int | None = None,
    dim1_offset: int | None = None,
):
  """Core bitonic sort implementation."""
  # Track global index for bitonic sort order (for array sub-sorting)
  # Second term controls whether final stage is descending or ascending
  dim1 = refs[0].shape[1]
  if log_n is None:
    log_n = log2(dim1)
  if dim1_offset is None:
    dim1_offset = (pl.program_id(1) * dim1 +
                   int(descending) * pl.num_programs(1) * dim1)

  if stage_ref is None:
    # Execute full bitonic sort
    _stages(
        1, log_n + 1, refs,
        num_keys=num_keys,
        dim1_offset=dim1_offset,
    )
  else:
    # Run single stage (for large arrays that don't fit in VMEM)
    stage = stage_ref[0]
    _stages(
        stage, stage + 1,
        refs,
        num_keys=num_keys,
        dim1_offset=dim1_offset,
        start_stage_static_lower_bound=log_n,
    )


def _sort_refs(
    in_refs,
    stage_ref,
    out_refs,
    refs, # scratch refs operated on
    indices_ref,
    *,
    descending: bool,
    is_stable: bool,
    num_keys: int,
    log_n: int | None = None,
):
  """Pallas kernel for sorting."""
  shape = in_refs[0].shape
  assert len(shape) == 2
  k = out_refs[0].shape[-1]

  if log_n is None:
    log_n = log2(shape[1])
  if 2**log2(shape[1]) != shape[1]:
    raise ValueError("Size along sort dimension must be a power of 2")

  return_argsort = len(out_refs) > len(in_refs)
  assert len(out_refs) == (len(in_refs) + int(return_argsort))

  use_indices = is_stable or return_argsort
  indices = indices_ref[...]

  if descending and is_stable:
    # Maintain order by sorting indices ascending while keys descending
    # Flip sign on indices, then flip back before write out
    indices = indices.shape[1] - indices

  # Reuse in/out VMEM buffers to reduce memory usage
  for i in range(len(in_refs)):
    if same_shape_dtype(in_refs[i], refs[i]):
      refs[i] = in_refs[i]
    else:
      refs[i][...] = in_refs[i][...].astype(refs[i].dtype)

  if jnp.issubdtype(refs[i].dtype, jnp.floating) and i < num_keys:
    f32_in_sortable_i32 = float_to_sortable_int(refs[i][...])
    refs[i] = refs[i].bitcast(jnp.int32)
    refs[i][...] = f32_in_sortable_i32

  if use_indices:
    if same_shape_dtype(indices_ref, out_refs[-1]):
      indices_ref = out_refs[-1]
    indices_ref[...] = indices
    refs.insert(num_keys, indices_ref)

  bitonic_sort(
      refs,
      stage_ref,
      descending=descending,
      num_keys=num_keys + int(is_stable),
      log_n=log_n,
  )

  if use_indices:
    refs.pop(num_keys)
  if return_argsort:
    if descending and is_stable:
      indices_ref[...] = indices_ref.shape[1] - indices_ref[...]
    refs.append(indices_ref)

  for ref, out_ref in zip(refs, out_refs, strict=True):
    if ref is not out_ref:
      out_ref[...] = ref[..., :k].astype(out_ref.dtype)


### VMEM-Based Sort (fits in VMEM)

@functools.partial(
    jit,
    static_argnames=("k", "block_token", "block_seq", "return_argsort",
                     "descending", "num_keys", "is_stable", "log_n", "interpret")
)
def _sort_in_vmem(
    operand: jax.Array | Sequence[jax.Array],
    num_keys: int,
    k: int | None = None,
    block_token: int | None = None,
    block_seq: int | None = None,
    return_argsort: bool = False,
    descending: bool = False,
    is_stable: bool = False,
    stage: int | None = None,
    log_n: int | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, ...]:
  """Sort arrays that fit in VMEM using Pallas.

  Args:
    operand: Input array(s) to sort (2D)
    num_keys: Number of arrays to use as sort keys
    k: Return only first k elements from sorted arrays
    block_token: Token blocking size for memory efficiency
    block_seq: Sequence blocking size for use if subsorting operands
    return_argsort: Whether to return argsort indices
    descending: Sort in descending order
    is_stable: Whether to perform stable sort
    stage: Specific stage to run (for multi-stage sorting)
    log_n: Length of sorted axis if array is padded

  Returns:
    Tuple of sorted arrays (and optionally argsort indices)
  """
  operands, shape = canonicalize_operand(operand)

  if k is None:
    k = shape[-1]
  if block_token is None:
    block_token = min(max(NUM_SUBLANES, (2**14) // shape[0]), shape[0])
  if block_seq is None:
    block_seq = shape[1]
  if k != shape[1] and block_seq != shape[1]:
    raise ValueError('k is not compatible with subsorting')

  block_shape = (block_token, block_seq)

  out_shapes = jax.tree.map(
      lambda v: jax.ShapeDtypeStruct((shape[0], k), v.dtype),
      tuple(operands)
  )
  if return_argsort:
    out_shapes += (jax.ShapeDtypeStruct((shape[0], k), jnp.int32),)

  in_specs = (
      [pl.BlockSpec(block_shape, lambda i, j: (i, j)) for _ in operands],
      pl.BlockSpec(memory_space=pltpu.SMEM) if stage is not None else None
  )
  out_specs = tuple(
      pl.BlockSpec((block_token, min(k, block_seq)), lambda i, j: (i, j))
      for _ in out_shapes
  )

  scratch_shapes = (
      [pltpu.VMEM(block_shape, to_32bit_dtype(ref.dtype)) for ref in operands],
      pltpu.VMEM(block_shape, jnp.int32),
  )

  if stage is not None:
    stage = stage[None]

  return pl.pallas_call(
      functools.partial(_sort_refs, descending=descending, num_keys=num_keys,
                        is_stable=is_stable, log_n=log_n),
      out_shape=(out_shapes,),
      in_specs=in_specs,
      out_specs=(out_specs,),
      scratch_shapes=scratch_shapes,
      grid=(shape[0] // block_token, shape[1] // block_seq),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=int(0.9 * 2**27),
      ),
      interpret=interpret,
  )(operands, stage)[0]


### HBM-Based Substage (for large arrays)

class _AsyncCopyGroup:
  """Bundles multiple async copy operations as single operation."""

  def __init__(self, copy_descriptors):
    self.copy_descriptors = tuple(copy_descriptors)

  def wait(self):
    """Wait for all copy operations to complete."""
    for descriptor in self.copy_descriptors:
      descriptor.wait()


def _substage_in_hbm_refs(
    input_hbm_refs,
    substage_ref,
    stage_ref,
    output_hbm_refs,
    input_semaphores,
    output_semaphores,
    input_vmem_refs,
    scratch_vmem_refs,
    output_vmem_refs,
    *,
    num_keys: int,
    descending: bool,
):
  """Kernel for substage that doesn't fit in VMEM."""
  shape = input_hbm_refs[0].shape
  # Handle sublane dimension indexing
  sublane_block = input_vmem_refs[0].shape[-2]
  sublane_slice = pl.dslice(pl.program_id(0) * sublane_block, sublane_block)
  input_hbm_refs, output_hbm_refs = jax.tree.map(
      lambda ref: ref.at[sublane_slice], (input_hbm_refs, output_hbm_refs)
  )

  substage = substage_ref[0]
  stage = stage_ref[0]
  slice_length = input_vmem_refs[0].shape[-1]
  pair_length = 2 ** (substage + 1)
  slices_per_pair = (pair_length // 2) // slice_length

  def _compute_pair_slice_start_index(i):
    pair_idx = i // slices_per_pair
    pair_subslice_idx = i % slices_per_pair
    return pair_idx * pair_length + pair_subslice_idx * slice_length

  def perform_dma(i, is_load):
    """Perform DMA operation (load or store)."""
    buffer_slot = lax.rem(i, 2)
    left_start = _compute_pair_slice_start_index(i)
    right_start = left_start + (pair_length // 2)
    sems = input_semaphores if is_load else output_semaphores
    copies = []

    for i_ref, (hbm_ref, vmem_ref) in enumerate(zip(
        *(input_hbm_refs, input_vmem_refs) if is_load
        else (output_hbm_refs, output_vmem_refs),
        strict=True
    )):
      for vmem_slot, start in enumerate((left_start, right_start)):
        # Tell compiler start indices are multiples of num_lanes
        start = pl.multiple_of(start, NUM_LANES)
        hbm_ref_slice = hbm_ref.at[:, pl.dslice(start, slice_length)]
        vmem_ref_slice = vmem_ref.at[buffer_slot, vmem_slot]
        sem = sems.at[buffer_slot, vmem_slot, i_ref]
        src, dst = ((hbm_ref_slice, vmem_ref_slice) if is_load
                    else (vmem_ref_slice, hbm_ref_slice))
        copies.append(
            pltpu.async_copy(src_ref=src, dst_ref=dst, sem=sem)
        )
    return _AsyncCopyGroup(copies)

  load_dma = functools.partial(perform_dma, is_load=True)
  store_dma = functools.partial(perform_dma, is_load=False)

  def compute(loop_idx):
    """Perform comparison and swap logic."""
    start_idx = _compute_pair_slice_start_index(loop_idx)
    slot = lax.rem(loop_idx, 2)

    refs = []
    for input_ref, scratch_ref in zip(input_vmem_refs, scratch_vmem_refs):
      if same_shape_dtype(input_ref, scratch_ref):
        refs.append(tuple(input_ref[slot]))
      else:
        scratch_ref[slot] = input_ref[slot].astype(scratch_ref.dtype)
        refs.append(tuple(scratch_ref[slot]))
    is_descending = create_bit_indicator(stage, start_idx + int(descending) * shape[1])
    outputs = compare_and_swap(
        *transpose_list_of_lists(refs),
        is_descending=is_descending,
        num_keys=num_keys
    )
    for (output_ref, (o_left, o_right)) in zip(output_vmem_refs, outputs):
      output_ref[slot, 0] = o_left.astype(output_ref.dtype)
      output_ref[slot, 1] = o_right.astype(output_ref.dtype)

  num_iterations = input_hbm_refs[0].shape[-1] // (2 * slice_length)
  assert num_iterations > 0

  # Pipeline: Load -> Compute -> Store
  initial_load = load_dma(0)
  if num_iterations > 1:
    next_load = load_dma(1)

  initial_load.wait()
  compute(0)

  if num_iterations == 1:
    store_dma(0).wait()
    return

  next_load.wait()

  @pl.loop(1, num_iterations - 1)
  def pipeline_iteration(loop_idx):
    store_op = store_dma(loop_idx - 1)
    load_op = load_dma(loop_idx + 1)
    compute(loop_idx)
    store_op.wait()
    load_op.wait()

  store_op = store_dma(num_iterations - 2)
  compute(num_iterations - 1)
  store_op.wait()
  store_dma(num_iterations - 1).wait()


@functools.partial(
    jax.jit,
    static_argnames=('block_shape', 'num_keys', 'descending', 'interpret')
)
def _substage_in_hbm(
    operand,
    substage,
    stage,
    num_keys: int,
    descending: bool,
    block_shape=None,
    interpret: bool = False,
):
  """Run substage without loading full lane dimension into VMEM."""
  operands, shape = canonicalize_operand(operand)
  if block_shape is None:
    block_shape = (NUM_SUBLANES, 2**(16 - log2(len(operands))))

  input_specs = (
      [pl.BlockSpec(memory_space=pltpu.ANY) for _ in operands],
      pl.BlockSpec(memory_space=pltpu.SMEM),
      pl.BlockSpec(memory_space=pltpu.SMEM),
  )

  output_shape = jax.tree.map(
      lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), tuple(operands)
  )
  num_refs = len(operands)
  input_vmems = jax.tree.map(
      lambda x: pltpu.VMEM((2, 2, *block_shape), x.dtype), operands
  )
  scratch_vmems = jax.tree.map(
      lambda x: pltpu.VMEM((2, 2, *block_shape), to_32bit_dtype(x.dtype)),
      operands
  )

  return pl.pallas_call(
      functools.partial(_substage_in_hbm_refs, num_keys=num_keys,
                        descending=descending),
      grid=(operands[0].shape[0] // block_shape[0],),
      out_shape=(output_shape,),
      in_specs=input_specs,
      out_specs=(tuple(input_specs[0]),),
      scratch_shapes=(
          pltpu.SemaphoreType.DMA((2, 2, num_refs)),
          pltpu.SemaphoreType.DMA((2, 2, num_refs)),
          input_vmems,
          scratch_vmems,
          input_vmems, # output_vmems
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=int(0.9 * 2**27)
      ),
      interpret=interpret,
  )(operands, substage[None], stage[None])[0]


### Public API

@functools.partial(
    jax.jit,
    static_argnames=('num_vmem_substages', 'descending', 'return_argsort',
                     'is_stable', 'num_keys', 'block_token', 'interpret')
)
def sort(
    operand: jax.Array | Sequence[jax.Array],
    num_keys: int,
    is_stable: bool = False,
    return_argsort: bool = False,
    descending: bool = False,
    num_vmem_substages: int | None = None,
    block_token: int | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, ...]:
  """Sort large arrays using hybrid HBM-VMEM approach.

  Handles arrays larger than VMEM by breaking into subsections, sorting in
  VMEM, then merging with HBM-based operations.

  Args:
    operand: Input array(s) to sort (2D or sequence of 2D arrays)
    num_keys: Number of arrays to use as sort keys (lexicographic order)
    is_stable: Whether to perform stable sort
    return_argsort: Whether to return argsort indices as last element
    descending: Sort in descending order
    num_vmem_substages: log2 of max size that fits in VMEM (auto-calculated)
    block_token: Token blocking size for memory efficiency

  Returns:
    Tuple of sorted arrays (and optionally argsort indices)
  """
  operands, shape = canonicalize_operand(operand)
  num_stages = log2(shape[1])

  if (shape[1] != 2**num_stages and
      any(not jnp.issubdtype(x.dtype, jnp.floating) for x in operands)):
    # If padded, integer/bool values in padding may leak unless stable
    # Floats handled by standardizing nans and padding with largest nan
    is_stable = True

  use_indices = return_argsort or is_stable
  if use_indices:
    indices = jax.lax.broadcasted_iota(jnp.int32, operands[0].shape, 1)
    if descending and is_stable:
      # Keys descending, but ties sorted ascending, so flip indices
      indices = shape[1] - indices
    indices_index = num_keys
    operands.insert(num_keys, indices)
    if is_stable:
      num_keys += 1

  if num_vmem_substages is None:
    # Heuristic to fit 128MB VMEM
    num_vmem_substages = 18 - log2(
        len(operands) + sum(not is_32bit(x) for x in operands) * 0.5
    )

  dtypes = [x.dtype for x in operands]

  # Optimize bf16 + u16 case by packing into single i32
  use_packed_bf16_u16 = (
      operands[0].dtype == jnp.bfloat16 and len(operands) == 2 and
      (operands[1].dtype == jnp.uint16 or
       (use_indices and shape[1] <= 2**16))
  )
  if use_packed_bf16_u16:
    operands = [pack_bf16_u16_to_i32(*operands)]
    num_keys = 1

  # Convert float keys to sortable int representation
  operands = [
      float_to_sortable_int(x)
      if jnp.issubdtype(x.dtype, jnp.floating) and i < num_keys
      else x
      for i, x in enumerate(operands)
  ]

  # Pad to required dimensions
  operands = [
      pad(x, block_shape=(NUM_SUBLANES, 'power_of_2_lanes'), prepend=(False, descending))
      for x in operands
  ]

  # Sort based on array size
  if num_stages <= num_vmem_substages:
    # Array fits in VMEM
    operands = _sort_in_vmem(
        operands,
        descending=descending,
        num_keys=num_keys,
        is_stable=False,
        return_argsort=False,
        block_token=block_token,
        log_n=num_stages,
        interpret=interpret
    )
  else:
    def _run_stage(stage, operands):
      """Execute complete sorting stage (HBM + VMEM)."""
      def _compute_substages_hbm_body(i, operands):
        substage = stage - 1 - i
        return _substage_in_hbm(
            operands, substage, stage, num_keys=num_keys, descending=descending,
            interpret=interpret
        )

      # HBM-based substages for cross-VMEM-block operations
      operands = jax.lax.fori_loop(
          0, stage - num_vmem_substages, _compute_substages_hbm_body, operands
      )

      # VMEM-based substages for within-block operations
      return _sort_in_vmem(
          operands,
          block_seq=2**num_vmem_substages,
          stage=stage,
          descending=descending,
          num_keys=num_keys,
          is_stable=False,
          interpret=interpret
      )

    # Initial bitonic sorting of VMEM-sized blocks
    operands = _sort_in_vmem(
        tuple(operands),
        block_seq=2**num_vmem_substages,
        stage=None,
        descending=descending,
        num_keys=num_keys,
        is_stable=False,
        interpret=interpret
    )

    # Merge blocks through successive stages
    operands = jax.lax.fori_loop(
        num_vmem_substages, num_stages + 1, _run_stage, operands
    )

  # Unpad
  if not descending:
    operands = tuple(x[:shape[0], :shape[1]] for x in operands)
  else:
    operands = tuple(x[:shape[0], -shape[1]:] for x in operands)

  # Unpack bf16-u16 if used
  if use_packed_bf16_u16:
    operands = unpack_bf16_u16_from_i32(operands[0])

  # Convert sortable ints back to floats
  operands = tuple(
      sortable_int_to_float(x)
      if (jnp.issubdtype(dtype, jnp.floating) and
          jnp.issubdtype(x.dtype, jnp.integer))
      else x
      for x, dtype in zip(operands, dtypes)
  )

  operands = list(operands)
  if use_indices:
    indices = operands.pop(indices_index)
    if return_argsort:
      if descending and is_stable:
        indices = shape[1] - indices
      operands.append(indices)

  return tuple(operands)


@functools.partial(
    jax.jit,
    static_argnames=('num_vmem_substages', 'descending', 'return_argsort',
                     'is_stable', 'num_keys', 'block_token', 'interpret')
)
def sort_xla_equivalent(
    operand,
    num_keys: int,
    is_stable: bool = False,
    return_argsort: bool = False,
    descending: bool = False,
    num_vmem_substages: int | None = None,
    block_token: int | None = None,
    interpret: bool | None = None,
) -> tuple[jax.Array, ...]:
  """Reference implementation using XLA sort for correctness testing."""
  del num_vmem_substages, block_token, interpret
  operands = jax.tree.leaves(operand)

  if return_argsort:
    operands.append(
        jax.lax.broadcasted_iota(jnp.int32, operands[0].shape, 1)
    )
  if descending and is_stable:
    operands.insert(
        num_keys,
        -jax.lax.broadcasted_iota(jnp.int32, operands[0].shape, 1)
    )
    num_keys += 1

  outs = jax.lax.sort(operands, num_keys=num_keys, is_stable=is_stable)

  if descending and is_stable:
    outs = list(outs)
    outs.pop(num_keys - 1)
  if descending:
    outs = tuple(x[..., ::-1] for x in outs)

  return tuple(outs)