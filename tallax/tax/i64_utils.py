"""High-precision i64 utilities for summation without overflow.

This module provides utilities for performing i64-precision summations on i32 arrays
by splitting values into smaller chunks and carefully tracking overflow across reductions.
"""

import jax
import jax.numpy as jnp


def i64_sum_dim1(x: jax.Array, chunk_size: int = 128):
  """Sum i32 array along dimension 1 with i64 precision using two-stage reduction.

  This function performs high-precision summation by splitting i32 values into
  16-bit chunks, summing them separately to avoid overflow, then harmonizing
  the results back into proper i64 representation.

  Algorithm:
  1. Split input into chunks of chunk_size along axis=1, padding last chunk with zeros if needed
  2. First stage: Split (n, m, chunk_size) into m tiles, split each i32 into
     upper/lower 16 bits, sum over m dimension (safe for m < 32k)
  3. Second stage: Split the i64 intermediate results again and sum over chunk_size
     dimension, harmonize back to final i64 result

  The key insight is that by splitting i32 values into 16-bit parts, we can safely
  sum up to 32k (2^15) values without overflow, since (2^16-1) * 2^15 < 2^31.

  Summation bounds:
  - First reduction: m < 32k tiles of size chunk_size, reduces from (n, total) to (n, chunk_size) i64
  - Second reduction: chunk_size < 32k values, reduces from (n, chunk_size) i64 to (n, 1) i64
  - Final result can represent sums up to (2^32-1) * 32k * chunk_size ≈ 2^57 (for chunk_size=128)

  Args:
    x: Input array of shape (n, total) with i32 dtype
    chunk_size: Size of chunks for splitting (default 128, must be < 32k)

  Returns:
    Tuple of (high_i32, low_i32) representing i64 sum of shape (n, 1)
    where value = high * 2^32 + low

  Constraints:
    - x.shape[1] must be < 2^31
    - x.ndim must be <= 2
    - chunk_size must be < 32768 (2^15) for safe summation
    - m = ceil(x.shape[1] / chunk_size) must be < 32768 (2^15) for safe summation

  Example:
    >>> x = jnp.arange(256, dtype=jnp.int32).reshape(2, 128)
    >>> high, low = i64_sum_dim1(x)
    >>> # high[0] * 2**32 + low[0] equals sum of first row
  """
  assert x.shape[1] < 2**31 and x.ndim <= 2
  assert chunk_size < 32768, "chunk_size must be < 32k (2^15) for safe summation"

  n = x.shape[0]
  total_len = x.shape[1]

  # Calculate number of full chunks and remainder
  m = total_len // chunk_size
  remainder = total_len % chunk_size

  # Bitmask for extracting lower 16 bits
  bitmask_16 = jnp.int32(2**16 - 1)

  # ============================================================================
  # FIRST REDUCTION: Sum over m dimension
  # ============================================================================

  # Split into chunks of size chunk_size along axis=1
  # Handle full chunks and remainder separately
  if remainder == 0:
    # All full chunks, use split directly
    chunks = jnp.split(x, m, axis=1)
  else:
    # Split full chunks and handle remainder separately
    full_chunk_len = m * chunk_size

    # Get full chunks
    if m > 0:
      full_chunks = jnp.split(x[:, :full_chunk_len], m, axis=1)
    else:
      full_chunks = []

    # Get remainder and pad it
    remainder_chunk = x[:, full_chunk_len:]
    pad_amount = chunk_size - remainder
    remainder_chunk_padded = jnp.pad(
      remainder_chunk,
      ((0, 0), (0, pad_amount)),
      mode='constant',
      constant_values=0
    )

    # Combine all chunks
    chunks = full_chunks + [remainder_chunk_padded]
    m += 1

  assert m < 32768, f"m={m} must be < 32k (2^15) for safe summation"

  # Stack chunks to create (n, m, chunk_size)
  x_stacked = jnp.stack(chunks, axis=1)

  # Split each i32 into upper 16 bits and lower 16 bits
  # For an i32 value v:
  #   upper = v >> 16      (extracts bits 16-31)
  #   lower = v & 0xFFFF   (extracts bits 0-15)
  upper = x_stacked >> 16
  lower = x_stacked & bitmask_16

  # Sum over m dimension (axis=1) -> (n, chunk_size)
  # Each sum can hold up to (2^16-1) * m where m < 2^15
  # Maximum: (2^16-1) * 2^15 = 2^31 - 2^15 < 2^31, safe for i32
  upper_sum = jnp.sum(upper, axis=1, dtype=jnp.int32)
  lower_sum = jnp.sum(lower, axis=1, dtype=jnp.int32)

  # Harmonize: convert (upper_sum, lower_sum) to proper i64 representation
  # The conceptual sum is: sum = (upper_sum << 16) + lower_sum
  # We need to extract this as (high_i32, low_i32) where value = high * 2^32 + low

  # Extract bits 16-31 from lower_sum (these need to be added to upper)
  overflow = lower_sum >> 16
  low_16 = lower_sum & bitmask_16  # Bits 0-15 of final value

  # Combine upper_sum (which represents bits 16-31 of original values) with overflow
  mid_32 = upper_sum + overflow  # Represents bits 16-47 of final value

  # Split mid_32 into final high and low parts
  high_i32 = mid_32 >> 16  # Bits 32-47 (stored in low 16 bits of i32)
  low_i32 = ((mid_32 & bitmask_16) << 16) | low_16  # Bits 0-31

  # ============================================================================
  # SECOND REDUCTION: Sum over chunk_size dimension
  # ============================================================================

  # We now have (n, chunk_size) i64 values represented as (high_i32, low_i32)
  # Split each i32 part into upper and lower 16 bits again

  high_upper = high_i32 >> 16
  high_lower = high_i32 & bitmask_16
  low_upper = low_i32 >> 16
  low_lower = low_i32 & bitmask_16

  # Sum over chunk_size dimension (axis=1) with keepdims=True -> (n, 1)
  # chunk_size < 2^15 so safe for i32 without overflow
  high_upper_sum = jnp.sum(high_upper, axis=1, keepdims=True, dtype=jnp.int32)
  high_lower_sum = jnp.sum(high_lower, axis=1, keepdims=True, dtype=jnp.int32)
  low_upper_sum = jnp.sum(low_upper, axis=1, keepdims=True, dtype=jnp.int32)
  low_lower_sum = jnp.sum(low_lower, axis=1, keepdims=True, dtype=jnp.int32)

  # Harmonize the 4 parts back to i64 representation (high_i32, low_i32)

  # Reconstruct low i32 part (bits 0-31 of final i64)
  low_overflow = low_lower_sum >> 16
  low_final_16 = low_lower_sum & bitmask_16
  low_mid = low_upper_sum + low_overflow
  low_overflow_to_high = low_mid >> 16  # Carry to high part
  low_final = ((low_mid & bitmask_16) << 16) | low_final_16

  # Reconstruct high i32 part (bits 32-63 of final i64)
  # First add carry from low part to high_lower
  high_lower_with_carry = high_lower_sum + low_overflow_to_high
  high_overflow = high_lower_with_carry >> 16
  high_final_16 = high_lower_with_carry & bitmask_16
  high_mid = high_upper_sum + high_overflow
  high_final = (high_mid << 16) | high_final_16

  return high_final, low_final
