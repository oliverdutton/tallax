"""
Int32-based sampling for numerically stable token sampling.

This module implements sampling using int32 weights instead of float probabilities
to avoid overflow and precision issues in cumulative sum computations.
"""

import math
import jax
import jax.numpy as jnp
from jax import lax
from jax.extend.random import threefry2x32_p

from tallax.tax.utils import NUM_LANES


def logits_to_int32_weights(logits, max_sum=None):
  """
  Convert logits to int32 weights for stable sampling.

  Args:
      logits: Logits array of shape (..., k)
      max_sum: Maximum allowed sum per batch (default: 2^30 to avoid overflow)

  Returns:
      int32_weights: Integer weights with same shape as logits
  """
  if max_sum is None:
    # Use 2^30 instead of 2^31-1 to leave headroom for cumsum operations
    max_sum = 1 << 30

  # Compute exp(logits - max) for numerical stability
  # Subtract max per batch (last dim)
  max_logits = jnp.max(logits, axis=-1, keepdims=True)
  exp_logits = jnp.exp(logits - max_logits)

  # Compute sum for normalization
  sum_exp = jnp.sum(exp_logits, axis=-1, keepdims=True)

  # Scale to int32 range: scale each batch to have sum ~= max_sum
  # This ensures we use the full int32 range while avoiding overflow
  scale = max_sum / sum_exp
  int32_weights = (exp_logits * scale).astype(jnp.int32)

  # Ensure no negative weights (shouldn't happen but be safe)
  int32_weights = jnp.maximum(int32_weights, 0)

  return int32_weights


def int32_cumsum(weights, axis=-1):
  """
  Compute cumulative sum in int32 without overflow.

  Args:
      weights: int32 array
      axis: Axis along which to compute cumsum

  Returns:
      Cumulative sum in int32
  """
  # JAX's cumsum can handle int32, but we ensure dtype preservation
  return jnp.cumsum(weights, axis=axis, dtype=jnp.int32)


def _find_boundary_chunk(
  cumsum_ref,
  target,
  k,
  chunk_size,
  ref_offset=None,
  active_chunk=None,
):
  """
  Find which chunk contains the k'th element matching target threshold.

  This implements one level of hierarchical boundary search by:
  1. Dividing the array into chunks of size chunk_size
  2. Computing cumsum within each chunk
  3. Finding which chunk contains the boundary

  Args:
      cumsum_ref: Cumulative sum array, shape (batch, total_k)
      target: Target threshold value to find, shape (batch, 1)
      k: Number of elements to find (for finding k'th match), shape (batch, 1)
      chunk_size: Size of each chunk
      ref_offset: Starting offset in the array (default: 0)
      active_chunk: Slice indicating active region (default: all)

  Returns:
      new_offset: Offset of the chunk containing boundary, shape (batch, 1)
      boundary_slice: Values in the boundary chunk, shape (batch, chunk_size)
      remaining_k: Remaining k to find within the chunk, shape (batch, 1)
  """
  batch_size = cumsum_ref.shape[0]
  total_k = cumsum_ref.shape[1]

  if ref_offset is None:
    ref_offset = jnp.zeros((batch_size, 1), dtype=jnp.int32)

  if active_chunk is None:
    active_chunk = cumsum_ref

  # Reshape into chunks: (batch, num_chunks, chunk_size)
  num_chunks = active_chunk.shape[1] // chunk_size
  chunked = active_chunk[:, :num_chunks * chunk_size].reshape(
    batch_size, num_chunks, chunk_size
  )

  # Count elements < target in each chunk
  below_target = (chunked < target[:, :, None]).astype(jnp.int32)
  chunk_counts = below_target.sum(axis=2)  # Shape: (batch, num_chunks)

  # Cumulative count across chunks to find which chunk contains k'th element
  cumulative_counts = jnp.cumsum(chunk_counts, axis=1)  # (batch, num_chunks)

  # Find first chunk where cumulative count >= k
  chunk_idx = (cumulative_counts < k).sum(axis=1, keepdims=True)  # (batch, 1)
  chunk_idx = jnp.clip(chunk_idx, 0, num_chunks - 1)

  # Extract the boundary chunk
  batch_indices = jnp.arange(batch_size)[:, None]
  boundary_slice = chunked[batch_indices, chunk_idx, :]  # (batch, 1, chunk_size)
  boundary_slice = boundary_slice.squeeze(1)  # (batch, chunk_size)

  # Calculate new offset
  new_offset = ref_offset + chunk_idx * chunk_size

  # Calculate remaining k within the chunk
  # Count how many elements were in previous chunks
  prev_count = jnp.where(
    chunk_idx > 0,
    cumulative_counts[batch_indices, chunk_idx - 1],
    jnp.zeros((batch_size, 1), dtype=jnp.int32)
  )
  remaining_k = k - prev_count

  return new_offset, boundary_slice, remaining_k


def int32_bsearch(batch_shape, predicate):
  """
  Batched binary search over int32 values for finding boundary indices.

  For each element of the batch, search for the largest int32 (closest to
  positive infinity) for which the predicate is False.

  Args:
      batch_shape: Shape of the search batch
      predicate: Monotonic function from int32 to bool. Returns False for all
                 numbers <= threshold, True for numbers > threshold.

  Returns:
      For each batch element, the largest int32 for which predicate is False.
  """
  current_bits = jnp.zeros(batch_shape, dtype=jnp.int32)

  # Bit 31 is special (sign bit) - it compares in opposite order
  midpoint = current_bits
  predicate_satisfied = predicate(midpoint)
  current_bits = current_bits | jnp.where(
    predicate_satisfied, jnp.uint32(1 << 31), jnp.uint32(0)
  )

  def loop_body(i, current_bits):
    bit_index = 30 - i
    bit = jnp.int32(1 << bit_index)
    midpoint = current_bits | bit
    predicate_satisfied = predicate(midpoint)
    current_bits = current_bits | jnp.where(
      predicate_satisfied, jnp.int32(0), bit
    )
    return current_bits

  current_bits = lax.fori_loop(0, 31, loop_body, current_bits)
  return current_bits


def find_boundary_idx(ref, k, threshold):
  """
  Find the index of the k'th element matching threshold using hierarchical search.

  Uses progressive refinement:
  1. Coarse search in large chunks
  2. Medium search in NUM_LANES chunks
  3. Fine search within tile using iota

  This avoids computing full cumsum and is optimized for TPU tile sizes.

  Args:
      ref: Cumulative sum array, shape (batch, total_k)
      k: Number of elements to find (for boundary), shape (batch, 1)
      threshold: Target threshold value, shape (batch, 1)

  Returns:
      boundary_idx: Index of the k'th element, shape (batch,)
  """
  assert ref.ndim == 2
  batch_size = ref.shape[0]
  total_k = ref.shape[1]

  # Phase 1: Coarse search - large chunks
  # For 262k dim1 -> use sqrt chunking for efficiency
  coarse_chunk_size = int(math.sqrt(total_k // NUM_LANES)) * NUM_LANES
  coarse_chunk_size = max(coarse_chunk_size, NUM_LANES)

  ref_offset, boundary_slice, k = _find_boundary_chunk(
    ref,
    target=threshold,
    k=k,
    chunk_size=coarse_chunk_size,
  )

  # Phase 2: Medium search - NUM_LANES chunks
  ref_offset, boundary_slice, k = _find_boundary_chunk(
    ref,
    target=threshold,
    k=k,
    chunk_size=NUM_LANES,
    ref_offset=ref_offset,
    active_chunk=boundary_slice,
  )

  # Phase 3: Within tile exact match using iota
  # For high parallelism, make 128 (batch, 1) tiles
  iota1 = lax.broadcasted_iota(jnp.int32, (batch_size, NUM_LANES), 1)

  # Find exact position within the boundary_slice
  below_threshold_mask = (boundary_slice < threshold).astype(jnp.int32)

  # Count matches at each position
  matches_at_position = []
  for i in range(NUM_LANES):
    count = (below_threshold_mask * (iota1 == i)).sum(1, keepdims=True)
    matches_at_position.append(count)

  # Stack and find cumulative matches
  matches = jnp.concatenate(matches_at_position, axis=1)  # (batch, NUM_LANES)
  cumulative_matches = jnp.cumsum(matches, axis=1)  # (batch, NUM_LANES)

  # Find first position where cumulative >= k
  within_tile_idx = (cumulative_matches < k).sum(axis=1, keepdims=True)
  within_tile_idx = jnp.clip(within_tile_idx, 0, NUM_LANES - 1)

  # Final boundary index
  boundary_idx = ref_offset + within_tile_idx
  boundary_idx = boundary_idx.squeeze(1)  # (batch,)

  return boundary_idx


def find_top_p_boundary_int32(cumsum_weights, total_weights, p):
  """
  Find the boundary index for top-p filtering using hierarchical chunk search.

  Hierarchical approach (avoids full cumsum scan):
  1. Compute threshold = p * total_weights
  2. Hierarchical search: coarse chunks → medium chunks → within tile
  3. Returns boundary index and cumsum value at boundary

  This is more efficient than full array scan for large k.

  Args:
      cumsum_weights: Cumulative sum of weights, shape (..., k)
      total_weights: Total sum of weights per batch, shape (..., 1)
      p: Top-p threshold(s), shape (...,) or scalar

  Returns:
      boundary_idx: Index of last token to include in top-p set, shape (...,)
      boundary_sum: Cumulative sum at boundary, shape (...,)
  """
  batch_size = cumsum_weights.shape[0]
  total_k = cumsum_weights.shape[-1]

  # Compute threshold as int32
  p_expanded = jnp.expand_dims(p, axis=-1) if p.ndim < cumsum_weights.ndim else p
  threshold = (p_expanded * total_weights).astype(jnp.int32)

  # We want to find how many elements are < threshold
  # This gives us the boundary index
  k = jnp.full((batch_size, 1), total_k, dtype=jnp.int32)  # Search all elements

  # Use hierarchical search
  boundary_idx = find_boundary_idx(cumsum_weights, k, threshold)

  # Clamp to valid range [0, total_k-1]
  boundary_idx = jnp.clip(boundary_idx, 0, total_k - 1)

  # Get the cumsum value at boundary
  batch_idx = jnp.arange(batch_size)
  boundary_sum = cumsum_weights[batch_idx, boundary_idx]

  return boundary_idx, boundary_sum


def sparse_random_int32(key_ref, indices, dim1_size, maxval):
  """
  Generate uniform random int32 in [0, maxval) for sparse indices.

  Args:
      key_ref: RNG key, shape (1, 2)
      indices: Tuple of index arrays (dim0_idx, dim1_idx)
      dim1_size: Size of the second dimension (for linearizing indices)
      maxval: Maximum value (exclusive), shape matching indices[0]

  Returns:
      Random int32 values in [0, maxval) with same shape as indices[0]
  """
  assert len(indices) == 2

  # Handle JAX key format
  if key_ref.ndim == 0:
    key_ref = jnp.reshape(jax.random.key_data(key_ref), (1, 2))

  # Generate random bits using Threefry2x32
  counts_lo = (indices[0] * dim1_size + indices[1]).astype(jnp.uint32)
  counts_hi = jnp.zeros_like(counts_lo)
  k1 = jnp.reshape(key_ref[0, 0], (1, 1))
  k2 = jnp.reshape(key_ref[0, 1], (1, 1))
  bits1, bits2 = threefry2x32_p.bind(k1, k2, counts_hi, counts_lo)
  bits = bits1 ^ bits2

  # Convert to float [0, 1) then scale to [0, maxval)
  # Use same approach as _bits_to_uniform but for int32 range
  float_bits = jax.lax.shift_right_logical(bits, jnp.uint32(9))  # Keep 23 bits
  one_bits = jnp.ones((), dtype=jnp.float32).view(jnp.uint32)
  float_bits = jax.lax.bitwise_or(float_bits, one_bits)
  uniform = jax.lax.bitcast_convert_type(float_bits, jnp.float32) - 1.0

  # Scale to [0, maxval) and convert to int32
  # Use floor to ensure we stay in [0, maxval)
  maxval_float = maxval.astype(jnp.float32)
  scaled = uniform * maxval_float
  return jnp.floor(scaled).astype(jnp.int32)


def sample_token_from_int32_cumsum(cumsum_weights, random_int, axis=-1):
  """
  Sample a token using hierarchical chunk search on int32 cumulative sum.

  Given cumsum and a random int in [0, total_sum), find the token index
  such that cumsum[idx-1] <= random_int < cumsum[idx].

  Hierarchical approach (avoids full array scan):
  1. Use random_int as threshold
  2. Hierarchical search: coarse chunks → medium chunks → within tile
  3. Find first index where cumsum > random_int

  This is more efficient than full array scan for large k.

  Args:
      cumsum_weights: Cumulative sum of weights, shape (..., k)
      random_int: Random integer in [0, total_sum), shape (...,)
      axis: Axis along which tokens are arranged (default: -1)

  Returns:
      token_idx: Selected token index, shape (...,)
  """
  if axis != -1:
    raise NotImplementedError("Only axis=-1 supported")

  batch_size = cumsum_weights.shape[0]
  k = cumsum_weights.shape[-1]

  # Expand random_int to (batch, 1) for threshold
  random_int_expanded = jnp.expand_dims(random_int, axis=-1)

  # We want to find first index where cumsum > random_int
  # This is equivalent to finding how many elements have cumsum <= random_int
  k_search = jnp.full((batch_size, 1), k, dtype=jnp.int32)

  # Use hierarchical search with random_int as threshold
  # Find boundary where cumsum transitions from <= random_int to > random_int
  token_idx = find_boundary_idx(cumsum_weights, k_search, random_int_expanded)

  # The result is the count of elements <= random_int
  # We want the first element > random_int, so this is already correct
  # But we need to add 1 if cumsum[token_idx] <= random_int
  batch_indices = jnp.arange(batch_size)
  at_boundary = cumsum_weights[batch_indices, token_idx]

  # If cumsum[token_idx] <= random_int, we need the next element
  token_idx = jnp.where(
    at_boundary <= random_int,
    token_idx + 1,
    token_idx
  )

  # Clamp to valid range [0, k-1]
  token_idx = jnp.clip(token_idx, 0, k - 1)

  return token_idx


def top_p_and_sample_int32(
  logits,
  indices,
  rng_key,
  top_p,
  axis=-1,
):
  """
  Complete top-p sampling using int32 arithmetic.

  Args:
      logits: Logits array, shape (..., k)
      indices: Token indices corresponding to logits, shape (..., k)
      rng_key: RNG key for sampling
      top_p: Top-p threshold, shape (...,) or scalar
      axis: Axis along which to sample

  Returns:
      sampled_token_idx: Selected token indices, shape (...,)
  """
  if axis != -1:
    raise NotImplementedError("Only axis=-1 supported")

  # Convert logits to int32 weights
  int32_weights = logits_to_int32_weights(logits)

  # Compute cumulative sum
  cumsum_weights = int32_cumsum(int32_weights, axis=-1)

  # Get total sum (last element of cumsum)
  total_weights = cumsum_weights[..., -1:]

  # Find top-p boundary
  boundary_idx, boundary_sum = find_top_p_boundary_int32(
    cumsum_weights, total_weights, top_p
  )

  # Apply top-p mask by creating a new cumsum up to boundary
  # For each batch, we want cumsum[0:boundary_idx+1]
  # Tokens beyond boundary are masked out

  # Create mask for valid tokens (within top-p)
  token_positions = jnp.arange(int32_weights.shape[-1])
  valid_mask = token_positions <= boundary_idx[..., None]

  # Masked cumsum (use full cumsum but we'll only look up to boundary)
  # Actually, we can just use the original cumsum and ensure we sample < boundary_sum

  # Generate random int in [0, boundary_sum) for each batch
  batch_size = logits.shape[0] if logits.ndim > 1 else 1
  if logits.ndim == 1:
    # Single batch case
    dim0_idx = jnp.array([0])
    dim1_idx = jnp.array([0])
  else:
    dim0_idx = jnp.arange(batch_size)
    dim1_idx = jnp.zeros(batch_size, dtype=jnp.int32)

  random_int = sparse_random_int32(
    rng_key,
    (dim0_idx, dim1_idx),
    dim1_size=1,  # Only one sample per batch
    maxval=boundary_sum,
  )

  # Sample token using binary search
  token_relative_idx = sample_token_from_int32_cumsum(cumsum_weights, random_int, axis=-1)

  # Map back to original token indices
  if logits.ndim == 1:
    sampled_token_idx = indices[token_relative_idx]
  else:
    batch_idx = jnp.arange(batch_size)
    sampled_token_idx = indices[batch_idx, token_relative_idx]

  return sampled_token_idx
