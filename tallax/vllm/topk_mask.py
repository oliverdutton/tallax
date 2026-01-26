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
import operator
import math
import jax
import jax.numpy as jnp
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
  # We don't have to slice out the remainder (or last chunk in general), it naturally gets handled
  num_chunks = arr.shape[1] // chunk_size
  chunks = [
    arr[:, i * chunk_size : (i + 1) * chunk_size].astype(jnp.float32)
    for i in range(num_chunks)
  ]
  assert chunk_size % NUM_LANES == 0
  target = pltpu.repeat(
    target,
    chunk_size // NUM_LANES,
    axis=1,
  )

  # Count matches in each chunk, keeping (batch, 1) shape
  num_matches = [
    (chunk == target).sum(axis=1, keepdims=True).astype(jnp.int32)
    for chunk in chunks
  ]

  # Build cumulative sums across chunks
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
  boundary_slices = [ref[:, pl.dslice(pl.multiple_of(ref_offset[i, 0], chunk_size), chunk_size)].astype(jnp.float32) for i in range(batch_size)]
  boundary_slice = boundary_slices[0]
  for i in range(1, batch_size):
    boundary_slice = jnp.where(iota0 == i, boundary_slices[i], boundary_slice)

  # Mask OOB indices to dtype min to ensure they don't interfere with comparisons
  if num_chunks * chunk_size != arr.shape[1]:
    boundary_slice = jnp.where(
      (ref_offset[:,:1] + iota1) < ref.shape[1],
      boundary_slice,
      get_dtype_info(boundary_slice).min
    )
  return ref_offset, boundary_slice, k

def find_boundary_idx(ref, k, threshold):
  """Find the index of the k'th element matching threshold."""

  assert ref.ndim == 2
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


@functools.partial(jax.jit, static_argnames=('compare_fn', 'num_parallel'))
def reduce_compare_sum(vals, threshold, compare_fn, num_parallel: int=7):
  """Computes (vals > threshold).sum(axis=1, keepdims=True) and other comparisons efficiently.

  For efficiency, threshold can be pre-broadcast as (b, NUM_LANES) or (b, 1)
  """
  if num_parallel == 0 or vals.shape[1] < NUM_LANES:
    assert threshold.shape[1] == 1
    return compare_fn(vals, threshold).sum(1, keepdims=True)
  chunks = [vals[:, i*NUM_LANES:(i+1)*NUM_LANES] for i in range(vals.shape[1] // NUM_LANES)]
  num_parallel = min(num_parallel, len(chunks)) # guard num_parallel
  if (has_remainder := (vals.shape[1] % NUM_LANES)):
    remainder_chunk_sum = compare_fn(
      vals[:, (vals.shape[1] // NUM_LANES) * NUM_LANES:],
      threshold[:,:1]
    ).sum(1, keepdims=True)

  partial_sums = [compare_fn(chunk, threshold).astype(jnp.int32) for chunk in chunks[:num_parallel]]
  for i, chunk in enumerate(chunks[num_parallel:]):
    partial_sums[i % num_parallel] += compare_fn(chunk, threshold)
  while len(partial_sums) > 1:
    # Reduce sum pairs
    n = len(partial_sums)
    for i in range(n // 2):
      partial_sums[i] += partial_sums[i + pl.cdiv(n, 2)]
    partial_sums = partial_sums[:pl.cdiv(n, 2)]
  full_sum = partial_sums[0].sum(1, keepdims=True)
  if has_remainder:
    full_sum += remainder_chunk_sum
  return full_sum


def topk_mask_ref_inputs(
  logits_ref,
  k_ref,
  *,
  replace_val: float,
  stable: bool,
  num_parallel: int,
):
  """Pallas kernel for topk masking with parallel chunk-based reduction.

  Args:
    logits_ref: Input logits reference [batch, vocab_size]
    k_ref: Number of top elements to keep [batch, 1]
    replace_val: Replacement value for masked elements
    stable: Whether to use stable masking
    num_parallel: Number of parallel chunks for reduction
  """

  # Step 1: Find k'th largest value
  logits = logits_ref[...].astype(jnp.float32)
  # Avoid broadcast in compare at the end of every search iter by pre-broadcasting to tile
  k = k_ref[...]
  # next value after the largest value where less than k gt it.
  bound_shape = (logits.shape[0], NUM_LANES if num_parallel != 0 else 1)
  k = jnp.broadcast_to(k, bound_shape)
  predicate_fn = lambda pivot: reduce_compare_sum(logits, pivot, operator.gt, num_parallel=num_parallel) < k
  finfo = jnp.finfo(logits_ref.dtype)
  _, threshold, _ = binary_search(
    predicate_fn,
    *(jnp.full(bound_shape, v, logits.dtype) for v in (finfo.min, finfo.max)),
    num_iter=logits_ref.dtype.itemsize * 8, # 32 for f32, 16 for bf16
  )

  assert logits.shape[1] % NUM_LANES == 0
  if not stable:
    # Simple threshold masking
    mask = (logits >= pltpu.repeat(
      threshold,
      logits.shape[1] // NUM_LANES,
      axis=1,
    ))
  else:
    # Stable masking, only k values
    # Find exact boundary for stable masking
    boundary_idx = find_boundary_idx(
      logits_ref,
      k=k - reduce_compare_sum(logits, threshold, operator.gt),
      threshold=threshold
    )
    threshold = pltpu.repeat(
      threshold,
      logits.shape[1] // NUM_LANES,
      axis=1,
    )
    boundary_idx = pltpu.repeat(
      boundary_idx,
      logits.shape[1] // NUM_LANES,
      axis=1
    )
    mask = (logits > threshold) | (
      (logits == threshold) &
      (jax.lax.broadcasted_iota(jnp.int32, logits_ref.shape, 1) <= boundary_idx)
    )
  return jnp.where(mask, logits, replace_val).astype(logits_ref.dtype)

def topk_mask_pallas_kernel(
  logits_ref,
  k_ref,
  output_ref,
  *,
  replace_val: float,
  stable: bool,
  num_parallel: int,
):
  output_ref[...] = topk_mask_ref_inputs(logits_ref, k_ref, replace_val=replace_val, stable=stable, num_parallel=num_parallel)


@functools.partial(
  jax.jit,
  static_argnames=["replace_val", "stable", "interpret", "num_parallel"]
)
def topk_mask_pallas(
  x: jax.Array,
  k: int,
  replace_val: float = -1e12,
  stable: bool = True,
  interpret: bool = False,
  num_parallel: int = 7,
) -> jax.Array:
  """Pallas-based topk mask with parallel chunk-based reduction.

  Args:
    x: Input array of shape [batch, vocab_size]
    k: Number of top elements
    replace_val: Value for masked elements
    stable: Whether to use stable masking
    interpret: Whether to use interpret mode
    num_parallel: Number of parallel chunks for reduction

  Returns:
    Masked array
  """
  batch_size, _vocab_size = x.shape
  k = jnp.broadcast_to(k, (batch_size, 1))
  output_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
  return pl.pallas_call(
    functools.partial(
      topk_mask_pallas_kernel,
      replace_val=replace_val,
      stable=stable,
      num_parallel=num_parallel,
    ),
    compiler_params=pltpu.CompilerParams(vmem_limit_bytes=int(0.9 * 2**27)),
    out_shape=output_shape,
    interpret=interpret,
  )(x, k)
