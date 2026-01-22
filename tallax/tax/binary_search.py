"""Binary search utilities for finding thresholds efficiently.

This module implements binary search using monotonic f32<->u32 conversions
for efficient searching in float32 space.
"""

import jax
import jax.numpy as jnp
from jax import lax


def monotonic_f32_to_u32(x: jax.Array) -> jax.Array:
  """Convert float32 to uint32 with monotonic ordering.

  Maps float32 values to uint32 bit patterns such that:
  - Larger floats map to larger uint32 values
  - The mapping is bijective for all finite float32 values
  - Useful for binary search over float32 values using integer arithmetic

  Args:
    x: float32 array

  Returns:
    uint32 array with monotonic ordering preserved
  """
  # Convert to bit pattern
  x_bits = lax.bitcast_convert_type(x, jnp.uint32)
  # Flip sign bit and all other bits if negative (to get correct ordering)
  sign_bit = jnp.uint32(1 << 31)
  is_negative = (x_bits & sign_bit) != 0
  # For negative numbers, flip all bits; for positive, flip just sign bit
  return jnp.where(is_negative, ~x_bits, x_bits ^ sign_bit)


def monotonic_u32_to_f32(x: jax.Array) -> jax.Array:
  """Convert uint32 back to float32, inverse of monotonic_f32_to_u32.

  Args:
    x: uint32 array from monotonic_f32_to_u32

  Returns:
    float32 array
  """
  sign_bit = jnp.uint32(1 << 31)
  # Check if this was originally negative (MSB is 0 after transformation)
  was_negative = (x & sign_bit) == 0
  # Reverse the transformation
  x_bits = jnp.where(was_negative, ~x, x ^ sign_bit)
  return lax.bitcast_convert_type(x_bits, jnp.float32)


def interp_f32(l: jax.Array, r: jax.Array) -> jax.Array:
  """Interpolate between two float32 values in the monotonic u32 space.

  Computes the midpoint in uint32 space (avoiding overflow) and converts back.

  Args:
    l: Left boundary (float32)
    r: Right boundary (float32)

  Returns:
    Midpoint value (float32)
  """
  # Convert to monotonic u32
  l_u32 = monotonic_f32_to_u32(l)
  r_u32 = monotonic_f32_to_u32(r)

  # Overflow-safe (l+r)//2 using the formula: (l//2) + (r//2) + ((l&1)+(r&1))//2
  one = jnp.uint32(1)
  m_u32 = (l_u32 // 2) + (r_u32 // 2) + ((l_u32 & one) + (r_u32 & one)) // 2

  return monotonic_u32_to_f32(m_u32)


def binary_search(
  x: jax.Array,
  predicate_fn,
) -> jax.Array:
  """Find threshold using binary search with custom predicate.

  Uses binary search in monotonic u32 space to efficiently find the threshold
  value. Binary search finds the LARGEST threshold where predicate is FALSE.

  Args:
    x: Input array of shape [..., vocab_size]
    predicate_fn: Function that takes (x, threshold) where threshold has shape
                  [..., 1] and returns boolean array of shape [..., 1]

  Returns:
    Threshold array of shape [...]
  """
  batch_shape = x.shape[:-1]

  # Binary search finds LARGEST value where predicate is FALSE

  def loop_body(state):
    l, r = state
    pivot = interp_f32(l, r)

    # Evaluate predicate at midpoint
    # pivot has shape (batch, 1), broadcasts correctly with x
    predicate_true = predicate_fn(x, pivot)

    # We want the largest value where predicate is FALSE
    # If predicate is TRUE at pivot, then the answer is < pivot, so update r = pivot
    # If predicate is FALSE at pivot, then the answer might be pivot or > pivot, so update l = pivot
    l = jnp.where(predicate_true, l, pivot)
    r = jnp.where(predicate_true, pivot, r)

    return (l, r)

  def cond(state):
    l, r = state
    # Continue while l and r are more than 1 ULP apart
    pivot = interp_f32(l, r)
    return jnp.any(pivot != l)

  # Initialize bounds with shape (batch, 1)
  l = jnp.full(batch_shape + (1,), -jnp.inf, dtype=x.dtype)
  r = jnp.full(batch_shape + (1,), jnp.inf, dtype=x.dtype)

  # Run binary search
  l, r = lax.while_loop(cond, loop_body, (l, r))

  # Return with shape (batch,)
  return l.squeeze(-1)
