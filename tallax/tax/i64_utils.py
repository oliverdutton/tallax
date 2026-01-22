"""High-precision i64 utilities for summation without overflow.

This module provides utilities for performing i64-precision summations on i32 arrays
by splitting values into smaller chunks and carefully tracking overflow across reductions.
"""

import jax
import jax.numpy as jnp


def i64_sum_dim1(x: jax.Array):
  """Sum i32 array along dimension 1 with i64 precision using two-stage reduction.

  This function performs high-precision summation by splitting i32 values into
  16-bit chunks, summing them separately to avoid overflow, then harmonizing
  the results back into proper i64 representation.

  Algorithm:
  1. First stage: Split (n, m*128) into m tiles of (n, 128), split each i32 into
     upper/lower 16 bits, sum over m dimension (safe for m < 32k)
  2. Second stage: Split the i64 intermediate results again and sum over 128
     dimension, harmonize back to final i64 result

  The key insight is that by splitting i32 values into 16-bit parts, we can safely
  sum up to 32k (2^15) values without overflow, since (2^16-1) * 2^15 < 2^31.

  Summation bounds:
  - First reduction: m < 32k tiles of size 128, reduces from (n, m*128) to (n, 128) i64
  - Second reduction: 128 < 32k values, reduces from (n, 128) i64 to (n, 1) i64
  - Final result can represent sums up to (2^32-1) * 32k * 128 ≈ 2^57

  Args:
    x: Input array of shape (n, m*128) with i32 dtype where m < 32k

  Returns:
    Tuple of (high_i32, low_i32) representing i64 sum of shape (n, 1)
    where value = high * 2^32 + low

  Constraints:
    - x.shape[1] must be < 2^31 and divisible by 128
    - x.ndim must be <= 2
    - m = x.shape[1] // 128 must be < 32768 (2^15) for safe summation

  Example:
    >>> x = jnp.arange(256, dtype=jnp.int32).reshape(2, 128)
    >>> high, low = i64_sum_dim1(x)
    >>> # high[0] * 2**32 + low[0] equals sum of first row
  """
  assert x.shape[1] < 2**31 and x.ndim <= 2

  n = x.shape[0]
  m = x.shape[1] // 128
  assert m < 32768, "m must be < 32k (2^15) for safe summation"
  assert x.shape[1] == m * 128, "shape[1] must be divisible by 128"

  # Bitmask for extracting lower 16 bits
  bitmask_16 = jnp.int32(2**16 - 1)

  # ============================================================================
  # FIRST REDUCTION: Sum over m dimension
  # ============================================================================

  # Reshape to (n, m, 128)
  x_reshaped = x.reshape(n, m, 128)

  # Split each i32 into upper 16 bits and lower 16 bits
  # For an i32 value v:
  #   upper = v >> 16      (extracts bits 16-31)
  #   lower = v & 0xFFFF   (extracts bits 0-15)
  upper = x_reshaped >> 16
  lower = x_reshaped & bitmask_16

  # Sum over m dimension (axis=1) -> (n, 128)
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
  # SECOND REDUCTION: Sum over 128 dimension
  # ============================================================================

  # We now have (n, 128) i64 values represented as (high_i32, low_i32)
  # Split each i32 part into upper and lower 16 bits again

  high_upper = high_i32 >> 16
  high_lower = high_i32 & bitmask_16
  low_upper = low_i32 >> 16
  low_lower = low_i32 & bitmask_16

  # Sum over 128 dimension (axis=1) with keepdims=True -> (n, 1)
  # 128 < 2^15 so safe for i32 without overflow
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
