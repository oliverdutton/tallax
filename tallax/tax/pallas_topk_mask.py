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

from tallax.tax.binary_search import binary_search
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
    logits_ref: Logits reference of shape [batch, vocab_size]
    target: Target value to match (shape [batch, 1])
    k: Target count (shape [batch, 1])
    chunk_size: Size of each chunk

  Returns:
    Tuple of (chunk_index, cumsum_before_chunk) where:
      - chunk_index: Index of chunk containing k'th match (shape [batch, 1])
      - cumsum_before_chunk: Cumulative matches before this chunk (shape [batch, 1])
  """
  active_chunk = ref if active_chunk is None else active_chunk
  # Calculate number of chunks using ceiling division
  num_chunks = pl.cdiv(active_chunk.shape[1], chunk_size)
  # If chunk size doesn't evenly divide array, we make the first chunk smaller. This choice avoids OOB indexing to extract chunk_size length slices later.
  chunk_offsets = [0] + sorted(filter(lambda x: x>0, [active_chunk.shape[1] - i * chunk_size for i in range(num_chunks)]))
  # Split into chunks (no padding needed, variable sizes OK)
  chunks = [active_chunk[:, lb:ub] for lb, ub in zip(chunk_offsets[:-1], chunk_offsets[1:], strict=True)]
  assert sum([c.shape[1] for c in chunks]) == active_chunk.shape[1]
  first_chunk_size = chunks[0].shape[1]
  assert len(set([c.shape[1] for c in chunks[1:]])) <= 1

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
  k -= sum((i == (boundary_idx - 1)) * c for i, c in enumerate(cumsums))

  # We'll do batch_size separate dslices into arr
  batch_size = ref.shape[0]
  iota0, iota1 = (jax.lax.broadcasted_iota(jnp.int32, (batch_size, chunk_size), dim) for dim in (0, 1))
  
  # first chunk may not be chunk_size in length if arr doesn't evenly divide
  ref_offset += jnp.maximum(first_chunk_size + (boundary_idx - 1) * chunk_size, 0)
  # We indead into ref instead of ref_slice as dynamic_slice not supported on arrays
  boundary_slices = [ref[:, pl.dslice(ref_offset[i, 0], chunk_size)] for i in range(batch_size)]
  boundary_slice = boundary_slices[0]
  for i in range(1, batch_size):
    boundary_slice = jnp.where(iota0 == i, boundary_slices[i], boundary_slice)
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

def topk_mask_pallas_kernel(
  logits_ref,
  output_ref,
  *,
  k: int,
  replace_val: float,
  stable: bool,
  chunk_size: int = 128,
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
  # next value after the largest value where less than k gt it.
  predicate_fn = lambda pivot: (logits > pivot).sum(-1, keepdims=True) < k
  bound_shape = (logits.shape[0], 1)
  _, threshold = binary_search(
    predicate_fn,
    *(jnp.full(bound_shape, v, logits.dtype) for v in (-jnp.inf, jnp.inf))
  )

  if not stable:
    # Simple threshold masking
    output_ref[...] = jnp.where(
      logits >= threshold,
      logits,
      replace_val
    )
    return

  # Step 2: Find exact boundary for stable masking
  boundary_idx = find_boundary_idx(
    logits_ref,
    k=k - (logits > threshold).sum(1, keepdims=True),
    threshold=threshold
  )
  mask = (logits > threshold) | (
    (logits == threshold) &
    (jax.lax.broadcasted_iota(jnp.int32, logits_ref.shape, 1) <= boundary_idx)
  )
  output_ref[...] = jnp.where(mask, logits, replace_val)


@functools.partial(
  jax.jit,
  static_argnames=["k", "replace_val", "stable", "interpret", "chunk_size"]
)
def topk_mask_pallas(
  x: jax.Array,
  k: int,
  replace_val: float = -1e12,
  stable: bool = True,
  interpret: bool = False,
  chunk_size: int = 128,
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

  # Ensure vocab_size is multiple of NUM_LANES
  if vocab_size % NUM_LANES != 0:
    pad_size = NUM_LANES - (vocab_size % NUM_LANES)
    x = jnp.pad(x, ((0, 0), (0, pad_size)), constant_values=-jnp.inf)
    padded = True
  else:
    padded = False

  output_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)

  result = pl.pallas_call(
    functools.partial(
      topk_mask_pallas_kernel,
      k=k,
      replace_val=replace_val,
      stable=stable,
      chunk_size=chunk_size,
    ),
    out_shape=output_shape,
    interpret=interpret,
  )(x)

  # Remove padding if added
  if padded:
    result = result[:, :vocab_size]

  return result
