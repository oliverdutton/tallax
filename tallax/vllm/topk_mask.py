"""Pallas kernel implementation of topk_mask with parallel chunk-based reduction.

This module implements an optimized topk_mask using Pallas with:
1. Binary search to find the k'th largest threshold value
2. Parallel chunk-based processing to find exact boundary for stable sorting
3. Fully unrolled operations with no loops or padding for TPU efficiency

The approach:
- Split vocabulary into fixed-size chunks
- Count matches in parallel across all chunks
- Build cumulative sums to find boundary chunk
- Use cumulative sum to find exact boundary index for stable top-k
"""

import functools
import math
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax.vllm.binary_search import binary_search
from tallax.tax.utils import NUM_LANES, get_dtype_info


def _find_boundary_chunk(
  ref,
  target,
  k,
  chunk_size: int,
  active_chunk: jax.Array | None = None,
  ref_offset: jax.Array | int = 0,
):
  """Find which chunk contains the k'th element matching target.

  Parallel approach: splits vocabulary into chunks, counts matches in each chunk,
  builds cumulative sums, and iterates to find which chunk contains
  the k'th matching element.

  Args:
    ref: Reference array of shape [batch, vocab_size]
    target: Target value to match (shape [batch, 1])
    k: Target count (shape [batch, 1])
    chunk_size: Size of each chunk
    active_chunk: Optional subset of ref to search in
    ref_offset: Offset into ref for indexing

  Returns:
    Tuple of (ref_offset, boundary_slice, k) where:
      - ref_offset: Updated offset to boundary chunk start
      - boundary_slice: Extracted boundary chunk [batch, chunk_size]
      - k: Updated k count after subtracting earlier chunks
  """
  arr = ref if active_chunk is None else active_chunk
  # Calculate number of chunks using ceiling division
  num_chunks = pl.cdiv(arr.shape[1], chunk_size)
  # Split into chunks with multiples of chunk_size (may be OOB for last chunk)
  chunks = [
    arr[:, i * chunk_size : (i + 1) * chunk_size]
    for i in range(num_chunks)
  ]

  # Count matches in each chunk, keeping (batch, 1) shape
  num_matches = [
    (chunk == target).sum(axis=1, keepdims=True).astype(jnp.int32)
    for chunk in chunks
  ]

  # Build cumulative sums across chunks (keep as list, no concatenate for TPU)
  cumsums = [num_matches[0]]
  for i in range(1, len(num_matches)):
    cumsums.append(cumsums[i - 1] + num_matches[i])

  boundary_idx = sum((c < k) for c in cumsums)
  # Subtract counts from all chunks BEFORE the boundary chunk
  k -= sum((i == (boundary_idx - 1)) * c for i, c in enumerate(cumsums))

  # We'll do batch_size separate dslices into arr
  batch_size = ref.shape[0]
  iota0, iota1 = (jax.lax.broadcasted_iota(jnp.int32, (batch_size, chunk_size), dim) for dim in (0, 1))

  # Update offset by multiples of chunk_size
  ref_offset += boundary_idx * chunk_size
  # Assure compiler offset is a multiple of chunk_size
  # This is us guaranteeing when using multiple iterations of find_boundary_chunk that current chunk_size evenly divides all previous chunk_sizes
  # Index into ref (not ref_slice) as dynamic_slice not supported on arrays
  # These dslices may be OOB, which is fine - we mask them out later
  boundary_slices = [ref[:, pl.dslice(pl.multiple_of(ref_offset[i, 0], chunk_size), chunk_size)] for i in range(batch_size)]
  boundary_slice = boundary_slices[0]
  for i in range(1, batch_size):
    boundary_slice = jnp.where(iota0 == i, boundary_slices[i], boundary_slice)

  # Mask OOB indices to dtype min to ensure they don't interfere with comparisons
  if num_chunks * chunk_size != arr.shape[1]:
    boundary_slice = jnp.where(
      (ref_offset + iota1) < ref.shape[1],
      boundary_slice,
      get_dtype_info(ref).min
    )
  return ref_offset, boundary_slice, k

def find_boundary_idx(ref, k, threshold):
  """Find the index of the k'th element matching threshold."""

  assert ref.ndim==2
  ref_offset, boundary_slice, k = _find_boundary_chunk(
    ref,
    target=threshold,
    k=k,
    # for 262k dim1 -> 2k tiles -> slow, so we do (45, 45) instead of 2048
    chunk_size=int(math.sqrt(ref.shape[1] // NUM_LANES)) * NUM_LANES,
  )
  ref_offset, boundary_slice, k = _find_boundary_chunk(
    ref,
    target=threshold,
    k=k,
    # for 262k -> 2k tiles, so we do (45, 45) instead of 2048
    chunk_size=NUM_LANES,
    ref_offset=ref_offset,
    active_chunk=boundary_slice
  )
  # Within tile cumsum check
  # For high parallelism we make 128 (b, 1) tiles instead of several rounds of cumsum on (b, 128)
  iota1 = jax.lax.broadcasted_iota(jnp.int32, (ref.shape[0], NUM_LANES), 1)
  num_matches = [(
    (boundary_slice == threshold).astype(jnp.int32) * (iota1 == i) 
    ).sum(1, keepdims=True) for i in range(NUM_LANES)]
  cumsums = [num_matches[0]]
  for i in range(1, len(num_matches)):
    cumsums.append(cumsums[i - 1] + num_matches[i])
  return (ref_offset + sum((c < k) for c in cumsums))  

def alu_minus_gt(lhs, rhs):
   # equiv to -(lhs > rhs).astype(jnp.int32), but avoids masks
   # Only valid if no NaN and no inf/-inf in values.
  # When lhs > rhs: rhs - lhs < 0 → sign bit = 1 → >> 31 gives -1 → negation gives 1 ✓
  # When lhs < rhs: rhs - lhs > 0 → sign bit = 0 → >> 31 gives 0 → negation gives 0 ✓
  # When lhs == rhs: rhs - lhs == 0 → sign bit = 0 → >> 31 gives 0 → negation gives 0 ✓
  assert lhs.dtype == jnp.float32
  return ((rhs - lhs).view(jnp.int32) >> 31)

def fast_sum(x, num_parallel=3):
  if num_parallel == 0:
    return x.sum(1, keepdims=True)
  running_sums = [
    x[:, i*NUM_LANES:(i+1)*NUM_LANES] for i in range(num_parallel)
  ]
  i = num_parallel
  while i * NUM_LANES < x.shape[1]:
    running_sums[i % num_parallel] += x[:, i*NUM_LANES:(i+1)*NUM_LANES]
    i += 1
  return sum(x for x in running_sums).sum(1, keepdims=True)


def topk_mask_ref_inputs(
  logits_ref,
  k_ref,
  *,
  replace_val: float,
  stable: bool,
  use_alu: bool,
  num_parallel: int,
):
  """Pallas kernel for topk masking with parallel chunk-based reduction.

  Args:
    logits_ref: Input logits reference [batch, vocab_size]
    output_ref: Output reference [batch, vocab_size]
    k: Number of top elements to keep (static)
    replace_val: Replacement value for masked elements
    stable: Whether to use stable masking
    chunk_size: Size of chunks for parallel reduction
  """

  # Step 1: Find k'th largest value
  logits = logits_ref[...]
  k = k_ref[...]
  # next value after the largest value where less than k gt it.
  if use_alu:
    print("Use ALU gt")
    # Use ALU as more vector registers than mask registers.
    predicate_fn = lambda pivot: (-fast_sum(alu_minus_gt(logits, pivot), num_parallel=num_parallel)) < k
  else:
    print("Use gt")
    predicate_fn = lambda pivot: fast_sum(logits > pivot, num_parallel=num_parallel) < k
  bound_shape = (logits.shape[0], 1)
  _, threshold = binary_search(
    predicate_fn,
    *(jnp.full(bound_shape, v, logits.dtype) for v in (-jnp.inf, jnp.inf))
  )

  if not stable:
    # Simple threshold masking
    return jnp.where(
      logits >= threshold,
      logits,
      replace_val
    )

  # Step 2: Find exact boundary for stable masking
  boundary_idx = find_boundary_idx(
    logits_ref,
    k=k - fast_sum(logits > threshold), #.sum(1, keepdims=True),
    threshold=threshold
  )
  mask = (logits > threshold) | (
    (logits == threshold) &
    (jax.lax.broadcasted_iota(jnp.int32, logits_ref.shape, 1) <= boundary_idx)
  )
  return jnp.where(mask, logits, replace_val)

def topk_mask_pallas_kernel(
  logits_ref,
  k_ref,
  output_ref,
  *,
  replace_val: float,
  stable: bool,
  use_alu: bool,
  num_parallel: int,
):
  output_ref[...] = topk_mask_ref_inputs(logits_ref, k_ref, replace_val=replace_val, stable=stable, use_alu=use_alu, num_parallel=num_parallel)


@functools.partial(
  jax.jit,
  static_argnames=["replace_val", "stable", "interpret", "use_alu", "num_parallel"]
)
def topk_mask_pallas(
  x: jax.Array,
  k: int,
  replace_val: float = -1e12,
  stable: bool = True,
  interpret: bool = False,
  use_alu: bool = False,
  num_parallel: int = 3,
) -> jax.Array:
  """Pallas-based topk mask with parallel chunk-based reduction.

  Args:
    x: Input array of shape [batch, vocab_size]
    k: Number of top elements
    replace_val: Value for masked elements
    stable: Whether to use stable masking
    interpret: Whether to use interpret mode
    chunk_size: Size of chunks for parallel reduction

  Returns:
    Masked array
  """
  batch_size, vocab_size = x.shape
  k = jnp.broadcast_to(k, (batch_size, 1))
  output_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
  return pl.pallas_call(
    functools.partial(
      topk_mask_pallas_kernel,
      replace_val=replace_val,
      stable=stable,
      use_alu=use_alu,
      num_parallel=num_parallel,
    ),
    compiler_params=pltpu.CompilerParams(vmem_limit_bytes=int(0.9 * 2**27)),
    out_shape=output_shape,
    interpret=interpret,
  )(x, k)
