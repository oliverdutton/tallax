"""
Int32-based sampling for numerically stable token sampling.

This module implements sampling using int32 weights instead of float probabilities
to avoid overflow and precision issues in cumulative sum computations.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.extend.random import threefry2x32_p


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


def find_top_p_boundary_int32(cumsum_weights, total_weights, p):
  """
  Find the boundary index for top-p filtering using binary search on int32 cumsum.

  Uses binary search to find the boundary index where cumulative sum reaches
  p * total_weights, providing better latency hiding on TPU compared to
  linear scan.

  Args:
      cumsum_weights: Cumulative sum of weights, shape (..., k)
      total_weights: Total sum of weights per batch, shape (..., 1)
      p: Top-p threshold(s), shape (...,) or scalar

  Returns:
      boundary_idx: Index of last token to include in top-p set, shape (...,)
      boundary_sum: Cumulative sum at boundary, shape (...,)
  """
  batch_shape = cumsum_weights.shape[:-1]
  k = cumsum_weights.shape[-1]

  # Compute threshold as int32
  p_expanded = jnp.expand_dims(p, axis=-1) if p.ndim < cumsum_weights.ndim else p
  threshold = (p_expanded * total_weights).astype(jnp.int32)
  threshold = threshold.squeeze(-1)  # Remove last dim to match batch_shape

  # Binary search predicate: Does cumsum[idx] >= threshold?
  # We want to find the largest idx where cumsum[idx] < threshold
  # which is the last token to include in top-p
  def predicate(idx):
    # Clamp idx to valid range [0, k-1]
    idx = jnp.clip(idx, 0, k - 1)
    # Get cumsum value at this index for each batch element
    batch_indices = jnp.arange(cumsum_weights.shape[0])
    cumsum_at_idx = cumsum_weights[batch_indices, idx]
    # Predicate is True when we've exceeded the threshold
    return cumsum_at_idx >= threshold

  # Binary search to find boundary index
  boundary_idx = int32_bsearch(batch_shape, predicate)

  # Clamp to valid range [0, k-1]
  boundary_idx = jnp.clip(boundary_idx, 0, k - 1)

  # Get the cumsum value at boundary (this is our new total for sampling)
  batch_idx = jnp.arange(cumsum_weights.shape[0])
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
  Sample a token using binary search on int32 cumulative sum.

  Given cumsum and a random int in [0, total_sum), find the token index
  such that cumsum[idx-1] <= random_int < cumsum[idx].

  Uses binary search for efficient token selection with latency hiding on TPU.

  Args:
      cumsum_weights: Cumulative sum of weights, shape (..., k)
      random_int: Random integer in [0, total_sum), shape (...,)
      axis: Axis along which tokens are arranged (default: -1)

  Returns:
      token_idx: Selected token index, shape (...,)
  """
  if axis != -1:
    raise NotImplementedError("Only axis=-1 supported")

  batch_shape = cumsum_weights.shape[:-1]
  k = cumsum_weights.shape[-1]

  # Binary search predicate: Does cumsum[idx] > random_int?
  # We want to find the smallest idx where cumsum[idx] > random_int
  # Which is equivalent to finding the largest idx where cumsum[idx] <= random_int
  # and then adding 1, but we'll use the predicate directly
  def predicate(idx):
    # Clamp idx to valid range [0, k-1]
    idx = jnp.clip(idx, 0, k - 1)
    # Get cumsum value at this index for each batch element
    batch_indices = jnp.arange(cumsum_weights.shape[0])
    cumsum_at_idx = cumsum_weights[batch_indices, idx]
    # Predicate is True when cumsum exceeds random_int
    return cumsum_at_idx > random_int

  # Binary search to find token index
  # This finds the largest index where predicate is False (cumsum <= random_int)
  token_idx = int32_bsearch(batch_shape, predicate)

  # The result is the largest idx where cumsum[idx] <= random_int
  # But we want the first idx where cumsum[idx] > random_int
  # So we need to add 1 and clamp
  token_idx = token_idx + 1

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
