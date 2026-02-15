"""Binary search utilities for finding thresholds efficiently.

This module implements binary search using monotonic f32<->u32 conversions
for efficient searching in float32 space.
"""

from collections.abc import Callable
import functools
import jax
import jax.numpy as jnp
from jax import lax


def monotonic_f32_to_u32(x: jax.Array) -> jax.Array:
  """Convert float32 to uint32 with monotonic ordering.

  Maps float32 values to uint32 bit patterns such that:
  - Larger floats map to larger uint32 values
  - The mapping is bijective for all finite float32 values

  Args:
    x: float32 array

  Returns:
    uint32 array with monotonic ordering preserved
  """
  x_bits = lax.bitcast_convert_type(x, jnp.uint32)
  sign_bit = jnp.uint32(1 << 31)
  is_negative = (x_bits & sign_bit) != 0
  return jnp.where(is_negative, ~x_bits, x_bits ^ sign_bit)


def monotonic_u32_to_f32(x: jax.Array) -> jax.Array:
  """Convert uint32 back to float32, inverse of monotonic_f32_to_u32.

  Args:
    x: uint32 array from monotonic_f32_to_u32

  Returns:
    float32 array
  """
  sign_bit = jnp.uint32(1 << 31)
  was_negative = (x & sign_bit) == 0
  x_bits = jnp.where(was_negative, ~x, x ^ sign_bit)
  return lax.bitcast_convert_type(x_bits, jnp.float32)


def _interp(l: jax.Array, r: jax.Array, underlying_dtype=None) -> jax.Array:
  """Interpolate between two values in monotonic u32 space (for floats) or directly (for ints).

  Computes the midpoint avoiding overflow.

  Args:
    l: Left boundary (float32 or int32)
    r: Right boundary (float32 or int32)
    underlying_dtype: If the values originate from a lower-precision dtype
      (e.g. bfloat16), snap the midpoint to that dtype's representable grid.

  Returns:
    Midpoint value (float32 or int32)
  """
  assert l.dtype in (jnp.float32, jnp.int32)
  floating = l.dtype == jnp.float32
  if floating:
    l = monotonic_f32_to_u32(l)
    r = monotonic_f32_to_u32(r)
  assert l.dtype in (jnp.uint32, jnp.int32)
  # Overflow-safe (l+r)//2
  one = jnp.full_like(l, 1)
  pivot = (l // 2) + (r // 2) + ((l & one) + (r & one)) // 2
  if floating:
    pivot = monotonic_u32_to_f32(pivot)
  if underlying_dtype is not None:
    pivot = pivot.astype(underlying_dtype).astype(pivot.dtype)
  return pivot


# Alias for backwards compatibility
interp_f32 = _interp


def binary_search(
  predicate_fn: Callable[[jax.Array], jax.Array],
  lower_bound: jax.Array = None,
  upper_bound: jax.Array = None,
  num_iter: int | None = None,
  underlying_dtype=None,
) -> jax.Array:
  """Find threshold using binary search with custom predicate.

  Binary search finds the LARGEST threshold where predicate is FALSE.
  Uses monotonic u32 space for float values.

  Args:
    predicate_fn: Function that takes a threshold and returns boolean array.
      Should return True when threshold is too low.
    lower_bound: Lower bound for search
    upper_bound: Upper bound for search
    num_iter: Number of iterations (fixed iteration count if given, else convergence-based)
    underlying_dtype: If searching over values from a lower-precision dtype,
      pass the original dtype for midpoint snapping.

  Returns:
    Tuple of (lower_bound, threshold, next_pivot) from final search state
  """
  interp = functools.partial(_interp, underlying_dtype=underlying_dtype)

  @jax.jit
  def loop_body(state):
    l, r, pivot = state

    # Pre-compute two possible pivots of next iter to reduce latency
    next_pivots = (interp(l, pivot), interp(pivot, r))

    predicate_true = predicate_fn(pivot)

    # Largest value where predicate is FALSE
    l = jnp.where(predicate_true, l, pivot)
    r = jnp.where(predicate_true, pivot, r)

    next_pivot = jnp.where(predicate_true, *next_pivots)
    return (l, r, next_pivot)

  def cond(state):
    l, _, next_pivot = state
    return jnp.any(next_pivot != l)

  state = (lower_bound, upper_bound, interp(lower_bound, upper_bound))
  if num_iter is not None:
    return jax.lax.fori_loop(0, num_iter, lambda _, carry: loop_body(carry), init_val=state)
  else:
    return lax.while_loop(cond, loop_body, state)
