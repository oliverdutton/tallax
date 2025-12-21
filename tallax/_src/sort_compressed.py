"""
Faster sorting implementation using compressed transpose format throughout.

This implementation minimizes transposes by:
1. Transposing to compressed format ONCE at load
2. Running all stages/substages in compressed format
3. Transposing back ONCE at store

This avoids the overhead of transpose/untranspose between stages.
"""

import functools

import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax._src.utils import (
    log2,
    max_int,
    pad,
    float_to_sortable_int,
    sortable_int_to_float,
    to_compressed_transpose_format,
    from_compressed_transpose_format,
    to_32bit_dtype,
    canonicalize_operand,
    NUM_LANES,
    NUM_SUBLANES,
)
from tallax._src.sort import (
    run_compressed_transpose_format_substages_on_tiles,
)


def _sort_in_vmem_compressed_refs(
    in_refs,
    out_refs,
    *,
    num_keys: int,
    descending: bool,
    unroll: int = 128,
):
  """Pallas kernel for sorting using compressed transpose format throughout.

  This implementation:
  1. Transposes input to compressed format ONCE at load
  2. Runs all stages and substages in compressed format
  3. Transposes back to normal format ONCE at store
  4. Uses compilation strategy to avoid too many tiles for large arrays

  Args:
    in_refs: Input references
    out_refs: Output references
    num_keys: Number of sort keys
    descending: Sort direction
    unroll: Number of tiles to unroll (default 128)
  """
  shape = in_refs[0].shape
  dim1 = shape[1]
  log_n = log2(dim1)

  # Load and transpose to compressed format ONCE
  arrs = [ref[...].astype(to_32bit_dtype(ref.dtype)) for ref in in_refs]

  # Convert floats to sortable int representation
  for i in range(num_keys):
    if jnp.issubdtype(in_refs[i].dtype, jnp.floating):
      arrs[i] = float_to_sortable_int(arrs[i])
      arrs[i] = arrs[i].bitcast(jnp.int32)

  batch_size = shape[0]

  # Pad to compressed transpose format requirements
  padded_arrs = [
      pad(arr, block_shape=(NUM_LANES * NUM_LANES // dim1, dim1))
      for arr in arrs
  ]
  batch_size = padded_arrs[0].shape[0]

  # Transpose to compressed format ONCE
  arrs_tiles = jax.tree.map(to_compressed_transpose_format, padded_arrs)

  num_tiles = len(arrs_tiles[0])

  # Calculate stage limits
  # unroll_stage_limit = stages we can fully unroll without compilation issues
  # For 128 tiles: log2(128 * 8 sublanes) = log2(1024) = 10
  unroll_stage_limit = log2(min(unroll * NUM_SUBLANES, num_tiles * NUM_SUBLANES))

  dim1_offset = int(descending) * dim1

  # Stages 1 to unroll_stage_limit: fully unrolled (sequential, not in a loop)
  # This allows the compiler to fuse operations
  for stage in range(1, min(unroll_stage_limit + 1, log_n + 1)):
    arrs_tiles = run_compressed_transpose_format_substages_on_tiles(
        arrs_tiles,
        num_substages=stage,
        stage=stage,
        batch_size=batch_size,
        num_keys=num_keys,
        dim1_offset=dim1_offset,
    )

  # For higher stages (if any), use fori_loop to avoid unrolling too much
  # Each high stage runs:
  #   1. High substages (> unroll_stage_limit) - these would cross unroll boundaries
  #   2. Low substages (<= unroll_stage_limit) - handled by existing compressed ops
  if log_n > unroll_stage_limit:
    @pl.loop(unroll_stage_limit + 1, log_n + 1)
    def run_high_stage(stage):
      # For high substages, we currently fall back to the existing implementation
      # which uses run_compressed_transpose_format_substages_on_tiles
      # This works but may have some transpose overhead for very high substages
      #
      # In the future, could optimize by implementing cross-unroll-boundary operations
      # directly in compressed format, but for now this works correctly
      nonlocal arrs_tiles
      arrs_tiles = run_compressed_transpose_format_substages_on_tiles(
          arrs_tiles,
          num_substages=min(log2(num_tiles * NUM_SUBLANES), stage),
          stage=stage,
          batch_size=batch_size,
          num_keys=num_keys,
          dim1_offset=dim1_offset,
      )

  # Transpose back from compressed format ONCE
  outs = [
      from_compressed_transpose_format(tiles, dim0=batch_size)[:shape[0]]
      for tiles in arrs_tiles
  ]

  # Convert back from sortable int representation
  for i in range(num_keys):
    if jnp.issubdtype(out_refs[i].dtype, jnp.floating):
      outs[i] = outs[i].bitcast(jnp.float32)
      outs[i] = sortable_int_to_float(outs[i])

  # Store results
  for out, out_ref in zip(outs, out_refs, strict=True):
    out_ref[...] = out.astype(out_ref.dtype)


@functools.partial(
    jit,
    static_argnames=("k", "descending", "num_keys", "unroll", "interpret")
)
def sort_compressed(
    operand: jax.Array | list[jax.Array],
    num_keys: int,
    k: int | None = None,
    descending: bool = False,
    unroll: int = 128,
    interpret: bool = False,
) -> tuple[jax.Array, ...]:
  """Sort arrays in VMEM using compressed transpose format throughout.

  This is a faster implementation that:
  - Transposes to compressed format once at start
  - Keeps data in compressed format throughout all stages
  - Avoids transpose overhead between stages/substages
  - Transposes back once at end
  - Uses controlled unrolling to avoid compilation issues with large arrays

  Example usage:
    # Sort (8, 2**13) array
    import jax.numpy as jnp
    x = jnp.arange(8 * 2**13, dtype=jnp.float32).reshape(8, 2**13)[:, ::-1]
    sorted_x, = sort_compressed(x, num_keys=1, descending=False)

  Args:
    operand: Input array(s) to sort (2D). Shape must be (batch, sort_dim)
             where batch <= NUM_LANES and sort_dim is power of 2.
    num_keys: Number of arrays to use as sort keys
    k: Return only first k elements (must equal full dim for now)
    descending: Sort in descending order
    unroll: Number of tiles to unroll (default 128 for good compile performance)
    interpret: Run in interpret mode

  Returns:
    Tuple of sorted arrays
  """
  operands, shape = canonicalize_operand(operand)

  if k is None:
    k = shape[-1]
  if k != shape[-1]:
    raise NotImplementedError("Top-k not yet implemented for compressed format sort")

  # Input must be power-of-2 in sort dimension
  if 2**log2(shape[1]) != shape[1]:
    raise ValueError(f"Sort dimension must be power of 2, got {shape[1]}")

  out_shapes = jax.tree.map(
      lambda v: jax.ShapeDtypeStruct(shape, v.dtype),
      tuple(operands)
  )

  return pl.pallas_call(
      functools.partial(
          _sort_in_vmem_compressed_refs,
          num_keys=num_keys,
          descending=descending,
          unroll=unroll,
      ),
      out_shape=(out_shapes,),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=int(0.9 * 2**27)
      ),
      interpret=interpret,
  )(operands)[0]
