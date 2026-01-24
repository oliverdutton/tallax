"""Parallel binary search utilities with multi-pivot evaluation.

This module extends the standard binary search to evaluate multiple pivot points
in parallel, reducing the number of iterations needed.
"""

from collections.abc import Callable
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


def interp(l: jax.Array, r: jax.Array) -> jax.Array:
  """Interpolate between two float32 values in the monotonic u32 space.

  Computes the midpoint in uint32 space (avoiding overflow) and converts back.

  Args:
    l: Left boundary (float32 or int32)
    r: Right boundary (float32 or int32)

  Returns:
    Midpoint value (float32 or int32)
  """
  assert l.dtype in (jnp.float32, jnp.int32)
  floating = l.dtype == jnp.float32
  if floating:
    l = monotonic_f32_to_u32(l)
    r = monotonic_f32_to_u32(r)
  assert l.dtype in (jnp.uint32, jnp.int32)
  # Overflow-safe (l+r)//2 using the formula: (l//2) + (r//2) + ((l&1)+(r&1))//2
  one = jnp.array(1, dtype=l.dtype)
  pivot = (l // 2) + (r // 2) + ((l & one) + (r & one)) // 2
  if floating:
    pivot = monotonic_u32_to_f32(pivot)
  return pivot


# Alias for backwards compatibility
interp_f32 = interp


def generate_pivots(l: jax.Array, r: jax.Array, num_pivots: int):
  """Generate num_pivots pivot points between l and r using recursive halving.

  Args:
    l: Left boundary
    r: Right boundary
    num_pivots: Number of pivots (must be 2^v - 1, e.g., 1, 3, 7, 15)

  Returns:
    List of pivot points in ascending order
  """
  assert num_pivots in (1, 3, 7, 15, 31, 63), f"num_pivots must be 2^v - 1, got {num_pivots}"

  if num_pivots == 1:
    return [interp(l, r)]

  # Compute center pivot
  mid = interp(l, r)

  if num_pivots == 3:
    # Split into 4 segments with 3 pivots
    left_mid = interp(l, mid)
    right_mid = interp(mid, r)
    return [left_mid, mid, right_mid]

  elif num_pivots == 7:
    # Recursively split: 3 pivots on left, 1 center, 3 pivots on right
    left_mid = interp(l, mid)
    right_mid = interp(mid, r)

    # Left 3 pivots
    ll = interp(l, left_mid)
    lr = interp(left_mid, mid)

    # Right 3 pivots
    rl = interp(mid, right_mid)
    rr = interp(right_mid, r)

    return [ll, left_mid, lr, mid, rl, right_mid, rr]

  elif num_pivots == 15:
    # Recursively split into 16 segments
    left_mid = interp(l, mid)
    right_mid = interp(mid, r)

    # Recursively generate 7 pivots on each half + center
    left_pivots = generate_pivots(l, mid, 7)
    right_pivots = generate_pivots(mid, r, 7)

    return left_pivots + right_pivots

  elif num_pivots in (31, 63):
    # For larger num_pivots, use recursive generation
    v = {31: 5, 63: 6}[num_pivots]
    left_pivots = generate_pivots(l, mid, 2**(v-1) - 1)
    right_pivots = generate_pivots(mid, r, 2**(v-1) - 1)
    return left_pivots + right_pivots

  else:
    raise ValueError(f"Unsupported num_pivots: {num_pivots}")


def binary_search(
  predicate_fn: Callable[[jax.Array], jax.Array],
  lower_bound: jax.Array = None,
  upper_bound: jax.Array = None,
  num_pivots: int = 1,
) -> jax.Array:
  """Find threshold using binary search with custom predicate.

  Uses binary search in monotonic u32 space to efficiently find the threshold
  value. Binary search finds the LARGEST threshold where predicate is FALSE.

  Args:
    predicate_fn: Function that takes threshold and returns boolean array
    lower_bound: Lower bound for search
    upper_bound: Upper bound for search
    num_pivots: Number of pivots to evaluate per iteration (1, 3, 7, 15, etc.)
                Must be 2^v - 1. Higher values reduce iterations but increase
                work per iteration.

  Returns:
    Tuple of (l, r) where l is largest value with predicate FALSE,
    r is smallest value with predicate TRUE
  """
  # Binary search finds LARGEST value where predicate is FALSE

  def loop_body(state):
    l, r = state

    # Multi-pivot evaluation
    pivots = generate_pivots(l, r, num_pivots)
    # Evaluate predicate at all pivots in parallel
    predicates = [predicate_fn(p) for p in pivots]
    for pivot, predicate_true in zip(pivots, predicates, strict=True):
      # If predicate is TRUE at pivot, answer is < pivot: update r = pivot
      # If predicate is FALSE at pivot, answer might be pivot or > pivot: update l = pivot
      l = jnp.where(predicate_true, l, pivot)
      r = jnp.where(predicate_true, pivot, r)
    return (l, r)

  def cond(state):
    l, r = state
    # Continue while l and r are more than 1 ULP apart
    pivot = interp(l, r)
    return jnp.any(pivot != l)

  # Run binary search, user decides if they need l or r
  return lax.while_loop(cond, loop_body, (lower_bound, upper_bound))
