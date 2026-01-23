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

from tallax.tax.utils import NUM_LANES, get_dtype_info


def find_boundary_chunk(
  ref,
  target,
  k,
  chunk_size: int,
  ref_slice: jax.Array | None = None,
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
  arr = ref if ref_slice is None else ref_slice

  # Calculate number of chunks using ceiling division
  num_chunks = pl.cdiv(arr.shape[1], chunk_size)
  # Split into chunks (no padding needed, variable sizes OK)
  chunks = [
    ref[:, i * chunk_size : min((i + 1) * chunk_size, arr.shape[1])]
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
  k -= sum((i < boundary_idx) * c for i, c in enumerate(cumsums)) # This should be >0

  # We'll do batch_size separate dslices into arr
  batch_size = ref.shape[0]
  iota0, iota1 = (jax.lax.broadcasted_iota(jnp.int32, (batch_size, chunk_size), dim) for dim in (0, 1))
  boundary_idx = boundary_idx.squeeze(1)
  # 8 memory accesses rather than num_chunks
  ref_offset += boundary_idx * chunk_size
  # We indead into ref instead of ref_slice as dynamic_slice not supported on arrays
  boundary_slices = [ref[:, pl.dslice(ref_offset[i], chunk_size)] for i in range(batch_size)]
  boundary_slice = boundary_slices[0]
  for i in range(1, batch_size):
    boundary_slice = jnp.where(iota0 == i, boundary_slices[i], boundary_slice)
  # mask to in range indexing
  boundary_slice = jnp.where(
    # index in bounds
    (boundary_idx * chunk_size + iota1) < ref.shape[1], boundary_slice, get_dtype_info(ref.dtype).min)

  return ref_offset, boundary_slice, k


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
  batch_size = logits_ref.shape[0]
  vocab_size = logits_ref.shape[1]

  # k is now a static int, create array for computations
  k_array = jnp.full((batch_size, 1), k, dtype=jnp.int32)

  # Step 1: Find threshold using top_k
  # Use lax.top_k to find the k'th largest value efficiently
  # This is simpler and more reliable than binary search in Pallas context
  threshold_vals, _ = lax.top_k(logits_ref[...], k)
  # Get the k'th value for each batch (last value in top_k result)
  threshold = threshold_vals[:, k - 1 : k]  # (batch, 1)

  if not stable:
    # Simple threshold masking
    output_ref[...] = jnp.where(
      logits_ref[...] >= threshold,
      logits_ref[...],
      replace_val
    )
    return

  # Step 2: Find exact boundary for stable masking
  # Two stages

  ref_offset, boundary_slice, k = find_boundary_chunk(
    logits_ref,
    target=threshold,
    k=k - (logits_ref[...] > threshold).sum(1, keepdims=True),
    # for 262k -> 2k tiles, so we do (45, 45) instead of 2048
    chunk_size=math.sqrt(logits_ref.shape[0] // NUM_LANES) * NUM_LANES,
  )
  ref_offset, boundary_slice, k = find_boundary_chunk(
    logits_ref,
    target=threshold,
    k=k,
    # for 262k -> 2k tiles, so we do (45, 45) instead of 2048
    chunk_size=NUM_LANES,
    ref_offset=ref_offset,
    ref_slice=boundary_slice
  )
  # Within tile cumsum check
  # For high parallelism we make 128 (b, 1) tiles instead of several rounds of cumsum on (b, 128)
  iota1 = jax.lax.broadcasted_iota(jnp.int32, (batch_size, NUM_LANES), 1)
  num_matches = [(
    (boundary_slice == threshold).astype(jnp.int32) * (iota1 == i) 
    ).sum(1, keepdims=True) for i in range(NUM_LANES)]
  cumsums = [num_matches[0]]
  for i in range(1, len(num_matches)):
    cumsums.append(cumsums[i - 1] + num_matches[i])
  boundary_idx = ref_offset + sum((c < k) for c in cumsums).squeeze(1)

  # Step 3: Apply mask using boundary index
  # Keep if value >= threshold AND index <= boundary_idx
  mask = (logits_ref[...] > threshold) | (
    logits_ref[...] == threshold &
    (jax.lax.broadcasted_iota(jnp.int32, logits_ref.shape, 1) <= boundary_idx)
  )
  output_ref[...] = jnp.where(mask, logits_ref[...], replace_val)


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
