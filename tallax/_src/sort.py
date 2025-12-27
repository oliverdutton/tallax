
import functools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax._src.utils import (
    log2,
    pad,
    float_to_sortable_int,
    sortable_int_to_float,
    pack_bf16_u16_to_i32,
    unpack_bf16_u16_from_i32,
    create_bit_indicator,
    transpose_list_of_lists,
    to_32bit_dtype,
    same_shape_dtype,
    canonicalize_operand,
    is_32bit,
    NUM_LANES,
    NUM_SUBLANES,
    ceil_multiple,
)
from tallax._src.bitonic_sort import (
    bitonic_sort_maybe_rolled,
    _compute_padded_shape as _compute_padded_shape_for_sort,
    compute_pair_slice_start_index as _compute_pair_slice_start_index,
    compare_and_swap,
)
from tallax._src.bitonic_topk import (
  bitonic_topk_arrays,
  _compute_padded_shape as _compute_padded_shape_for_topk
)
from tallax._src.symint import SymInt


### VMEM-Based Sort (fits in VMEM)
def bitonic_sort_in_vmem_refs(
    in_refs,
    stage_ref,
    out_refs,
    transpose_refs,  # scratch refs operated on
    tranpose_indices_ref,
    *,
    descending: bool,
    is_stable: bool,
    num_keys: int,
    k: int | None = None,
    stage_unroll: int | None = None,
    slice_size_unroll: int | None = None,
    ref_slice_size_unroll: int | None = None,
    unroll_stages: bool = True,
    # int key operands may contain INT_MAX (ascending sort) or INT_MIN (descending sort) requiring stable sort to avoid padding leakage
    int_key_operands_may_contain_int_max_or_min: bool = True,
):
  """Pallas kernel for sorting using bitonic sort."""
  operands = [ref[...] for ref in in_refs]
  shape = operands[0].shape
  assert len(shape) == 2
  padded_shape = _compute_padded_shape_for_sort(*shape) if k!=shape[1] else _compute_padded_shape_for_topk(*shape, k=k)
  
  return_argsort = len(out_refs) > len(in_refs)
  assert len(out_refs) == (len(in_refs) + int(return_argsort))

  if int_key_operands_may_contain_int_max_or_min and (shape[1] != padded_shape[1]) and any(not jnp.issubdtype(x.dtype, jnp.floating) for x in operands):
    # Floats handled by standardizing 'in bounds' nans to a lower bit value to padding nans in the i32 format theyre sorted in 
    # But integer values of INT_MAX/INT_MIN in padding may leak unless comparison is stable
    is_stable = True

  use_indices = is_stable or return_argsort
  indices = jax.lax.broadcasted_iota(jnp.int32, shape, 1)

  if descending and is_stable:
    # Maintain order by sorting indices ascending while keys descending
    # Reverse indices (negate relative to array length), then reverse back before write out
    indices = indices.shape[1] - 1 - indices

  # Optimize bf16 + u16 case by packing into single i32
  use_packed_bf16_u16 = (
      len(operands) == 2 and operands[0].dtype == jnp.bfloat16 and 
      (operands[1].dtype == jnp.uint16 or
       (use_indices and shape[1] <= 2**16))
  )
  if use_packed_bf16_u16:
    operands = [pack_bf16_u16_to_i32(*operands)]
    transpose_refs = transpose_refs[:1]
    num_keys = 1
    is_stable = False # itll be stable by default
  
  for i in range(num_keys):
    if jnp.issubdtype(operands[i].dtype, jnp.floating):
      operands[i] = float_to_sortable_int(operands[i])    

  if use_indices:
    operands.insert(num_keys, indices)
    transpose_refs.insert(num_keys, tranpose_indices_ref)
  num_keys += int(is_stable)
  
  # Most TPU generations only allow 32,32->1 bit comparisons,
  # not bf16,bf16->i1 so we upcast everything to 32bit
  operands = [x.astype(to_32bit_dtype(x.dtype)) for x in operands]

  # Compute sort_dim_offset outside of run_scoped to avoid grid context issues
  sort_dim_offset = (
      # local
      SymInt(pl.program_id(1), 0, pl.num_programs(1)-1) * shape[1] +
      # global
      int(descending) * pl.num_programs(1) * shape[1])
  if k == shape[1]:
    # sort
    operands = bitonic_sort_maybe_rolled(
        num_keys=num_keys,
        axis=1,
        descending=False, # handled already
        single_stage=None if stage_ref is None else stage_ref[0],
        stage_unroll=stage_unroll,
        slice_size_unroll=slice_size_unroll,
        ref_slice_size_unroll=ref_slice_size_unroll,
        unroll_stages=unroll_stages if stage_ref is None else False,
        # only used if unroll_stages, then this ceases to be an _arrays method
        transpose_refs=transpose_refs,
        sort_dim_offset=sort_dim_offset
    )
  else:
    # top-k
    # this is fully unrolled and supports only certain k
    def _maybe_invert(operands):
      if not descending:
        for i in range(num_keys):
          # all keys should be ints, we flip them
          assert jnp.isdtype(operands[i].dtype, 'integral')
          operands[i] = jnp.invert(operands[i])

    _maybe_invert(operands)
    operands = bitonic_topk_arrays(operands=operands, k=k, num_keys=num_keys, axis=1)
    _maybe_invert(operands)
    
  # Unpack bf16-u16 if used
  if use_packed_bf16_u16:
    operands = unpack_bf16_u16_from_i32(operands[0])

  if use_indices:
    indices = operands.pop(num_keys - int(is_stable))
  if return_argsort:
    if descending and is_stable:
      indices = indices.shape[1] - 1 - indices
    operands.append(indices)

  for i, (out, out_ref) in enumerate(zip(operands, out_refs, strict=True)):
    if jnp.issubdtype(out.dtype, jnp.integer) and jnp.issubdtype(out_ref.dtype, jnp.floating):
      # Check if this was a float key that we converted
      out = sortable_int_to_float(out)
    out_ref[...] = out.astype(out_ref.dtype)


@functools.partial(
    jit,
    static_argnames=("num_keys", "return_argsort", "descending", "is_stable", "k",
                     "num_stages", "interpret", "block_token", "block_seq",
                     "compile_fast", "stage_unroll", "slice_size_unroll",
                     "ref_slice_size_unroll", "unroll_stages")
)
def bitonic_sort_in_vmem(
    operand: jax.Array | Sequence[jax.Array],
    # behavior control
    num_keys: int,
    return_argsort: bool = False,
    descending: bool = False,
    is_stable: bool = False,
    k: int | None = None,
    # niche behavior for larger than vmem inputs
    stage: int | jax.Array | None = None,
    interpret: bool = False,
    # implementation details
    block_token: int | None = None,
    block_seq: int | None = None,

    compile_fast: bool | None = None,
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
  operands, shape = canonicalize_operand(operand)
  
  if stage_unroll is None:
    # heuristic, likely reduces register pressure as it reorder operations into groups of 2**6/NUM_SUBLANES=8 tiles
    stage_unroll = 6
  if compile_fast is None:
    # if projected compilation time expected to be more than a minute, compile fast
    compile_fast = (shape[0] * shape[1] * (len(operands) + int(return_argsort or is_stable))) > 2**19
  if compile_fast:
    # reduces compilation time scaling to linear
    stage_unroll, slice_size_unroll, ref_slice_size_unroll, unroll_stages = (6, 7, 8, False)

  unconverted_operands = tuple(operands)
  # On CPU (interpret mode), convert floats to sortable ints outside Pallas to avoid ref bitcast lowering issues. On TPU, keep conversion inside Pallas kernel for efficiency (and it can allow bf16-u16 packing)
  if interpret:
    for i in range(num_keys):
      if jnp.issubdtype(operands[i].dtype, jnp.floating):
        operands[i] = float_to_sortable_int(operands[i])

  if block_token is None:
    # heuristic for fitting in VMEM
    # multiple of NUM_SUBLANES, between NUM_SUBLANES and NUM_LANES
    block_token = min(
      ceil_multiple(
        min((2**14) // shape[0], shape[0]), NUM_SUBLANES), NUM_LANES)
  if block_seq is None:
    block_seq = shape[1]
  if shape[1] % block_seq != 0:
    raise ValueError
  if k is None:
    k = shape[1]
  if k != shape[1] and block_seq != shape[1]:
    raise ValueError('k is not compatible with subsorting')

  block_shape = (block_token, block_seq)

  out_shapes = jax.tree.map(
      lambda v: jax.ShapeDtypeStruct((shape[0], k), v.dtype),
      unconverted_operands
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

  # Create transpose scratch refs for bitonic sort
  dim0, dim1 = _compute_padded_shape_for_sort(*block_shape)
  transpose_block_shape = (dim1 // pl.cdiv(NUM_LANES, dim0), NUM_LANES)
  scratch_shapes = (
      [pltpu.VMEM(transpose_block_shape, jnp.int32 if i < num_keys else to_32bit_dtype(ref.dtype)) for i, ref in operands],
      pltpu.VMEM(transpose_block_shape, jnp.int32),
  )

  if stage is not None:
    stage = stage[None]

  return pl.pallas_call(
      functools.partial(bitonic_sort_in_vmem_refs, descending=descending, num_keys=num_keys,
                        is_stable=is_stable,
                        k=k,
                        stage_unroll=stage_unroll,
                        slice_size_unroll=slice_size_unroll,
                        ref_slice_size_unroll=ref_slice_size_unroll,
                        unroll_stages=unroll_stages,
                        ),
      out_shape=(out_shapes,),
      in_specs=in_specs,
      out_specs=(out_specs,),
      scratch_shapes=scratch_shapes,
      grid=(pl.cdiv(shape[0], block_token), shape[1] // block_seq),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=int(0.9 * 2**27),
      ),
      interpret=interpret,
  )(operands, stage)[0]
  

@functools.partial(
    jit,
    static_argnames=("k", "num_keys", "return_argsort", "descending", "is_stable", "k",
                     "interpret", "block_token")
)
def bitonic_topk_in_vmem(
    operand: jax.Array | Sequence[jax.Array],
    k: int,
    num_keys: int = 1,
    return_argsort: bool = True,
    descending: bool = True,
    is_stable: bool = False,
    block_token: int | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, ...]:
  return bitonic_sort_in_vmem(operand, num_keys=num_keys, return_argsort=return_argsort, descending=descending, is_stable=is_stable, k=k, block_token=block_token, interpret=interpret)


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
def bitonic_sort_large_shapes(
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
    operands = bitonic_sort_in_vmem(
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
      return bitonic_sort_in_vmem(
          operands,
          block_seq=2**num_vmem_substages,
          stage=stage,
          descending=descending,
          num_keys=num_keys,
          is_stable=False,
          interpret=interpret
      )

    # Initial bitonic sorting of VMEM-sized blocks
    operands = bitonic_sort_in_vmem(
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
