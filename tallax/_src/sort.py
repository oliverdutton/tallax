
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
    to_compressed_transpose_format,
    from_compressed_transpose_format,
    transpose_list_of_lists,
    to_32bit_dtype,
    same_shape_dtype,
    canonicalize_operand,
    is_32bit,
    NUM_LANES,
    NUM_SUBLANES,
)
from tallax._src.bitonic_sort import (
    bitonic_sort_arrays,
    _compute_padded_shape as _bitonic_compute_padded_shape,
    compute_pair_slice_start_index as _compute_pair_slice_start_index,
    compare_and_swap,
)
from tallax._src.symint import SymInt


### VMEM-Based Sort (fits in VMEM)
def _sort_in_vmem_bitonic_refs(
    in_refs,
    stage_ref,
    out_refs,
    refs,  # scratch refs operated on
    indices_ref,
    *,
    descending: bool,
    is_stable: bool,
    num_keys: int,
    num_stages: int | None = None,
    stage_unroll: int | None = None,
    slice_size_unroll: int | None = None,
    ref_slice_size_unroll: int | None = None,
    unroll_stages: bool = True,
    float_keys_converted_outside: list[bool] | None = None,
):
  """Pallas kernel for sorting using bitonic sort."""
  shape = in_refs[0].shape
  assert len(shape) == 2
  k = out_refs[0].shape[-1]

  if 2**log2(shape[1]) != shape[1]:
    raise ValueError("Size along sort dimension must be a power of 2")

  return_argsort = len(out_refs) > len(in_refs)
  assert len(out_refs) == (len(in_refs) + int(return_argsort))

  use_indices = is_stable or return_argsort
  indices = indices_ref[...]

  if descending and is_stable:
    # Maintain order by sorting indices ascending while keys descending
    # Reverse indices (negate relative to array length), then reverse back before write out
    indices = indices.shape[1] - 1 - indices

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

  # Use bitonic sort instead of _run_stages
  # Create transpose refs for bitonic sort
  dim0, dim1 = _bitonic_compute_padded_shape(*refs[0].shape, k=NUM_SUBLANES)
  dim0 = min(dim0, NUM_LANES)
  transpose_shape = (dim1 // (NUM_LANES // dim0), NUM_LANES)

  # Compute sort_dim_offset outside of run_scoped to avoid grid context issues
  sort_dim_offset = (
      # local
      SymInt(pl.program_id(1), 0, pl.num_programs(1)-1) * shape[1] +
      # global
      int(descending) * pl.num_programs(1) * shape[1])

  @functools.partial(pl.run_scoped, transpose_refs=[
      pltpu.VMEM(transpose_shape, to_32bit_dtype(ref.dtype)) for ref in refs
  ])
  def _run_bitonic(transpose_refs):
    outs = bitonic_sort_arrays(
        [ref[...] for ref in refs],
        num_keys=num_keys + int(is_stable),
        axis=1,
        descending=descending,
        single_stage=stage_ref[0] if stage_ref is not None else None,
        num_stages=num_stages,
        stage_unroll=stage_unroll,
        slice_size_unroll=slice_size_unroll,
        ref_slice_size_unroll=ref_slice_size_unroll,
        unroll_stages=unroll_stages if stage_ref is None else False,
        # only used if unroll_stages, then this ceases to be an _arrays method
        transpose_refs=transpose_refs,
        sort_dim_offset=sort_dim_offset
    )

    if use_indices:
      indices = outs.pop(num_keys)
    if return_argsort:
      if descending and is_stable:
        indices = indices.shape[1] - 1 - indices
      refs.append(indices)
  
    for i, (out, out_ref) in enumerate(zip(outs, out_refs, strict=True)):
      if jnp.issubdtype(out.dtype, jnp.integer) and jnp.issubdtype(out_ref.dtype, jnp.floating):
        # Check if this was a float key that we converted
        out = sortable_int_to_float(out)
      out_ref[...] = out.astype(out_ref.dtype)


@functools.partial(
    jit,
    static_argnames=("num_keys", "return_argsort", "descending", "is_stable",
                     "num_stages", "interpret", "block_token", "block_seq",
                     "compile_fast", "stage_unroll", "slice_size_unroll",
                     "ref_slice_size_unroll", "unroll_stages")
)
def _sort_in_vmem_bitonic(
    operand: jax.Array | Sequence[jax.Array],
    # behavior control
    num_keys: int,
    return_argsort: bool = False,
    descending: bool = False,
    is_stable: bool = False,
    # niche behavior for larger than vmem inputs
    stage: int | jax.Array | None = None,
    num_stages: int | None = None,
    interpret: bool = False,
    # implementation details
    block_token: int | None = None,
    block_seq: int | None = None,

    compile_fast: bool = False,
    # specialist unroll controls, suggest setting just fast_compile=True if compilation is too slow, it will overwrite and set these other unrolls
    stage_unroll: int | None = None,
    slice_size_unroll: int | None = None,
    ref_slice_size_unroll: int | None = None,
    unroll_stages: bool = True,
) -> tuple[jax.Array, ...]:
  """Sort arrays that fit in VMEM using bitonic sort.

  Args:
    operand: Input array(s) to sort (2D)
    num_keys: Number of arrays to use as sort keys
    return_argsort: Whether to return argsort indices
    descending: Sort in descending order
    is_stable: Whether to perform stable sort
    stage: Specific stage to run (for multi-stage sorting)
    num_stages: Number of stages in the bitonic sort
    interpret: Run in interpret mode
    block_token: Token blocking size for memory efficiency
    block_seq: Sequence blocking size for use if subsorting operands
    compile_fast: Use faster compilation settings (reduced unrolling)
    stage_unroll: Number of stages to unroll in bitonic sort
    slice_size_unroll: Slice size unroll parameter for bitonic sort
    ref_slice_size_unroll: Ref slice size unroll parameter for bitonic sort
    unroll_stages: Whether to unroll stages in bitonic sort

  Returns:
    Tuple of sorted arrays (and optionally argsort indices)
  """
  if stage_unroll is None:
    # heuristic, likely reduces register pressure as it reorder operations into groups of 2**6/NUM_SUBLANES=8 tiles
    stage_unroll = 6
  if compile_fast:
    # reduces compilation time scaling to linear
    stage_unroll, slice_size_unroll, ref_slice_size_unroll, unroll_stages = (6, 7, 8, False)
      
  operands, shape = canonicalize_operand(operand)
  k = shape[1]  # For compatibility with block_seq checks

  unconverted_operands = tuple(operands)
  # On CPU (interpret mode), convert floats to sortable ints outside Pallas to avoid ref bitcast lowering issues. On TPU, keep conversion inside Pallas kernel for efficiency
  if interpret:
    for i in range(len(operands)):
      if jnp.issubdtype(operands[i].dtype, jnp.floating) and i < num_keys:
        operands[i] = float_to_sortable_int(operands[i])

  if block_token is None:
    block_token = min(max(NUM_SUBLANES, (2**14) // shape[0]), shape[0])
  if block_seq is None:
    block_seq = shape[1]
  if k != shape[1] and block_seq != shape[1]:
    raise ValueError('k is not compatible with subsorting')

  block_shape = (block_token, block_seq)

  out_shapes = jax.tree.map(
      lambda v: jax.ShapeDtypeStruct(shape, v.dtype),
      unconverted_operands
  )
  if return_argsort:
    out_shapes += (jax.ShapeDtypeStruct(shape, jnp.int32),)

  in_specs = (
      [pl.BlockSpec(block_shape, lambda i, j: (i, j)) for _ in operands],
      pl.BlockSpec(memory_space=pltpu.SMEM) if stage is not None else None
  )
  out_specs = tuple(
      pl.BlockSpec((block_token, block_seq), lambda i, j: (i, j))
      for _ in out_shapes
  )

  # Allocate scratch refs with int32 for float keys to avoid dtype conversion issues
  # If float was already converted outside, it's already int32
  scratch_shapes = (
      [pltpu.VMEM(block_shape, to_32bit_dtype(ref.dtype)) for ref in operands],
      pltpu.VMEM(block_shape, jnp.int32),
  )

  if stage is not None:
    stage = stage[None]

  return pl.pallas_call(
      functools.partial(_sort_in_vmem_bitonic_refs, descending=descending, num_keys=num_keys,
                        is_stable=is_stable, num_stages=num_stages,
                        stage_unroll=stage_unroll,
                        slice_size_unroll=slice_size_unroll,
                        ref_slice_size_unroll=ref_slice_size_unroll,
                        unroll_stages=unroll_stages,
                        ),
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


def _run_array_substage_on_hbm_refs(
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

  def perform_dma(i, is_load):
    """Perform DMA operation (load or store)."""
    buffer_slot = lax.rem(i, 2)
    left_start = _compute_pair_slice_start_index(i, separation=pair_length, slice_length=slice_length)
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
def _run_array_substage_in_hbm(
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
      functools.partial(_run_array_substage_on_hbm_refs, num_keys=num_keys,
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

  if any(jnp.isdtype(x.dtype, 'bool') for x in operands):
    raise NotImplementedError('Please cast bool operands to integer')

  if (shape[1] != 2**num_stages and
      any(not jnp.issubdtype(x.dtype, jnp.floating) for x in operands)):
    # If padded, integer values in padding may leak unless stable
    # Floats handled by standardizing nans and padding with largest nan
    is_stable = True

  use_indices = return_argsort or is_stable
  if use_indices:
    indices = jax.lax.broadcasted_iota(jnp.int32, operands[0].shape, 1)
    if descending and is_stable:
      # Keys descending, but ties sorted ascending, so reverse indices
      indices = shape[1] - 1 - indices
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
    operands = _sort_in_vmem_bitonic(
        operands,
        descending=descending,
        num_keys=num_keys,
        is_stable=False,
        return_argsort=False,
        block_token=block_token,
        num_stages=num_stages,
        interpret=interpret,
    )
  else:
    def _run_stage(stage, operands):
      """Execute complete sorting stage (HBM + VMEM)."""
      def _compute_substages_hbm_body(i, operands):
        substage = stage - 1 - i
        return _run_array_substage_in_hbm(
            operands, substage, stage, num_keys=num_keys, descending=descending,
            interpret=interpret
        )

      # HBM-based substages for cross-VMEM-block operations
      operands = jax.lax.fori_loop(
          0, stage - num_vmem_substages, _compute_substages_hbm_body, operands
      )

      # VMEM-based substages for within-block operations
      return _sort_in_vmem_bitonic(
          operands,
          block_seq=2**num_vmem_substages,
          stage=stage,
          descending=descending,
          num_keys=num_keys,
          is_stable=False,
          interpret=interpret
      )

    # Initial bitonic sorting of VMEM-sized blocks
    operands = _sort_in_vmem_bitonic(
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
        indices = shape[1] - 1 - indices
      operands.append(indices)

  return tuple(operands)


@functools.partial(
    jax.jit,
    static_argnames=('num_vmem_substages', 'descending', 'return_argsort',
                     'is_stable', 'num_keys', 'block_token', 'interpret')
)
def xla_equivalent_sort(
    operand,
    num_keys: int,
    is_stable: bool = False,
    return_argsort: bool = False,
    descending: bool = False,
    num_vmem_substages: int | None = None,
    block_token: int | None = None,
    interpret: bool | None = None,
) -> tuple[jax.Array, ...]:
  """Reference implementation using XLA sort for correctness testing.

  Args:
    operand: Input array(s) to sort
    num_keys: Number of sort keys
    is_stable: Whether to perform stable sort
    return_argsort: Whether to return argsort indices
    descending: Sort in descending order
    num_vmem_substages: Ignored (compatibility arg)
    block_token: Ignored (compatibility arg)
    interpret: Ignored (compatibility arg)

  Returns:
    Tuple of sorted arrays (and optionally argsort indices)
  """
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


# Re-export functions for backwards compatibility with modules that import from sort
compute_pair_slice_start_index = _compute_pair_slice_start_index
# compare_and_swap is already imported without renaming, so it's available
# _run_compressed_transpose_format_substage_on_tiles is not in the new implementation