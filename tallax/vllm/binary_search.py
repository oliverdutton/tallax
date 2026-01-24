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


def find_new_bounds_unrolled(pivots, predicates):
  """Find new [l, r] bounds using unrolled scan over pivot evaluations.

  Given pivots and their predicate values, find the tightest bounds where:
  - l is the largest pivot where predicate is FALSE (or original l if all TRUE)
  - r is the smallest pivot where predicate is TRUE (or original r if all FALSE)

  Args:
    pivots: List of pivot values in ascending order
    predicates: List of boolean predicate evaluations at each pivot

  Returns:
    (new_l, new_r) tuple
  """
  num_pivots = len(pivots)

  if num_pivots == 1:
    # Simple case: one pivot
    pred = predicates[0]
    # If predicate is TRUE, answer is < pivot (r = pivot, l unchanged)
    # If predicate is FALSE, answer is >= pivot (l = pivot, r unchanged)
    return pivots[0], pred

  elif num_pivots == 3:
    # Unrolled scan for 3 pivots
    # Find the transition point where predicate switches from FALSE to TRUE

    # Build decision logic:
    # If all FALSE: l = pivots[2], keep r
    # If predicates = [FALSE, FALSE, TRUE]: l = pivots[1], r = pivots[2]
    # If predicates = [FALSE, TRUE, *]: l = pivots[0], r = pivots[1]
    # If all TRUE: keep l, r = pivots[0]

    p0, p1, p2 = predicates

    # Find rightmost FALSE pivot (becomes new l)
    # Start from left, accumulate
    new_l_candidates = [
      (True, pivots[2]),           # If all FALSE: use pivots[2]
      (~p2, pivots[1]),             # If p2 is FALSE: use pivots[1]
      (~p1 & p2, pivots[0]),        # If p1 FALSE, p2 TRUE: use pivots[0]
      (~p0 & p1, pivots[0]),        # Should not reach (covered above)
    ]

    # Find leftmost TRUE pivot (becomes new r)
    new_r_candidates = [
      (p0, pivots[0]),              # If p0 TRUE: use pivots[0]
      (~p0 & p1, pivots[1]),        # If p0 FALSE, p1 TRUE: use pivots[1]
      (~p1 & p2, pivots[2]),        # If p0,p1 FALSE, p2 TRUE: use pivots[2]
    ]

    # Select using cascading where
    new_l_idx = (~p2).astype(jnp.int32) * 2 + (~p1 & p2).astype(jnp.int32)
    new_r_idx = p0.astype(jnp.int32) * 0 + (~p0 & p1).astype(jnp.int32) * 1 + (~p0 & ~p1 & p2).astype(jnp.int32) * 2

    # Build arrays and index
    pivot_array = jnp.stack(pivots)
    pred_array = jnp.stack(predicates)

    # Find transition: last FALSE and first TRUE
    # If all FALSE: return last pivot, and indicate "no upper bound found"
    # If all TRUE: return "no lower bound found", and first pivot

    any_true = jnp.any(pred_array)
    any_false = jnp.any(~pred_array)

    # Find index of last FALSE (for new_l)
    # Scan from right: find last occurrence of FALSE
    false_indices = jnp.where(~pred_array, jnp.arange(num_pivots), -1)
    last_false_idx = jnp.max(false_indices)

    # Find index of first TRUE (for new_r)
    true_indices = jnp.where(pred_array, jnp.arange(num_pivots), num_pivots)
    first_true_idx = jnp.min(true_indices)

    new_l_value = pivot_array[last_false_idx] if any_false else pivots[0]
    new_r_value = pivot_array[first_true_idx] if any_true else pivots[-1]

    return new_l_value, new_r_value, any_false, any_true

  elif num_pivots == 7:
    # Unrolled scan for 7 pivots
    pred_array = jnp.stack(predicates)
    pivot_array = jnp.stack(pivots)

    any_true = jnp.any(pred_array)
    any_false = jnp.any(~pred_array)

    # Find last FALSE and first TRUE
    false_indices = jnp.where(~pred_array, jnp.arange(num_pivots), -1)
    last_false_idx = jnp.max(false_indices)

    true_indices = jnp.where(pred_array, jnp.arange(num_pivots), num_pivots)
    first_true_idx = jnp.min(true_indices)

    new_l_value = pivot_array[last_false_idx] if any_false else pivots[0]
    new_r_value = pivot_array[first_true_idx] if any_true else pivots[-1]

    return new_l_value, new_r_value, any_false, any_true

  else:
    # General case for larger num_pivots
    pred_array = jnp.stack(predicates)
    pivot_array = jnp.stack(pivots)

    any_true = jnp.any(pred_array)
    any_false = jnp.any(~pred_array)

    false_indices = jnp.where(~pred_array, jnp.arange(num_pivots), -1)
    last_false_idx = jnp.max(false_indices)

    true_indices = jnp.where(pred_array, jnp.arange(num_pivots), num_pivots)
    first_true_idx = jnp.min(true_indices)

    # Safe indexing with clip
    last_false_idx = jnp.clip(last_false_idx, 0, num_pivots - 1)
    first_true_idx = jnp.clip(first_true_idx, 0, num_pivots - 1)

    new_l_value = jnp.where(any_false, pivot_array[last_false_idx], pivots[0])
    new_r_value = jnp.where(any_true, pivot_array[first_true_idx], pivots[-1])

    return new_l_value, new_r_value, any_false, any_true


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

    if num_pivots == 1:
      # Original single-pivot behavior
      pivot = interp(l, r)
      predicate_true = predicate_fn(pivot)

      # If predicate is TRUE at pivot, answer is < pivot: update r = pivot
      # If predicate is FALSE at pivot, answer might be pivot or > pivot: update l = pivot
      l = jnp.where(predicate_true, l, pivot)
      r = jnp.where(predicate_true, pivot, r)

      return (l, r)

    else:
      # Multi-pivot evaluation
      pivots = generate_pivots(l, r, num_pivots)

      # Evaluate predicate at all pivots in parallel
      predicates = [predicate_fn(p) for p in pivots]

      # Find new bounds using unrolled scan
      new_l, new_r, any_false, any_true = find_new_bounds_unrolled(pivots, predicates)

      # Update bounds
      # If any_false, update l; otherwise keep l
      # If any_true, update r; otherwise keep r
      l = jnp.where(any_false, new_l, l)
      r = jnp.where(any_true, new_r, r)

      return (l, r)

  def cond(state):
    l, r = state
    # Continue while l and r are more than 1 ULP apart
    pivot = interp(l, r)
    return jnp.any(pivot != l)

  # Run binary search, user decides if they need l or r
  return lax.while_loop(cond, loop_body, (lower_bound, upper_bound))
