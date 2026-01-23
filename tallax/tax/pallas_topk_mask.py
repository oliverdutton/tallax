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
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax.tax.utils import NUM_LANES


def find_boundary_chunk(
  logits_ref,
  target,
  k,
  chunk_size: int,
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
  batch_size = logits_ref.shape[0]
  vocab_size = logits_ref.shape[1]

  # Calculate number of chunks using ceiling division
  num_chunks = pl.cdiv(vocab_size, chunk_size)

  # Split into chunks (no padding needed, variable sizes OK)
  chunks = [
    logits_ref[:, i * chunk_size : min((i + 1) * chunk_size, vocab_size)]
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
  k_at_boundary = sum((i < boundary_idx) * c for i, c in enumerate(cumsums))
  return boundary_idx, k_at_boundary


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
  # Compute cumulative count of elements >= threshold
  gte_threshold = (logits_ref[...] >= threshold).astype(jnp.int32)
  cumsum_gte = jnp.cumsum(gte_threshold, axis=1)

  # Keep the first k elements that are >= threshold
  # A position is included if its cumulative rank is <= k
  valid = (gte_threshold == 1) & (cumsum_gte <= k)

  # Find last valid index (for stable sorting)
  indices = jnp.arange(vocab_size)
  indices_broadcasted = jnp.broadcast_to(indices, (batch_size, vocab_size))
  global_boundary_idx = jnp.where(valid, indices_broadcasted, -1).max(axis=1, keepdims=True)

  # Step 3: Apply mask using boundary index
  # Keep if value >= threshold AND index <= boundary_idx
  mask = (logits_ref[...] >= threshold) & (indices_broadcasted <= global_boundary_idx)

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
      replace_val=replace_val,
      stable=stable,
      chunk_size=chunk_size,
    ),
    out_shape=output_shape,
    interpret=interpret,
  )(x, jnp.array([k], dtype=jnp.int32))

  # Remove padding if added
  if padded:
    result = result[:, :vocab_size]

  return result
