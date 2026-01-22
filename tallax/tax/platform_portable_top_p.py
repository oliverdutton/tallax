"""Platform-portable top-p sampling using high-precision integer arithmetic.

This module implements a top-p (nucleus sampling) implementation that is:
1. Platform-portable: gives identical results across different hardware
2. Summation-order invariant: results don't depend on floating-point summation order
3. High-precision: uses safe integer arithmetic with 16-bit parts

The algorithm:
1. Convert logits to unnormalized probabilities: exp(logits - max(logits))
2. Scale f32 probabilities to i32 range [0, 2^30]
3. Sum using high-precision arithmetic (16-bit parts, sum max 2^15 values)
4. Binary search for threshold where cumulative sum >= top_p * total_sum
"""

import jax
import jax.numpy as jnp
from jax import lax

from tallax.tax.high_precision_uint import HighPrecisionUInt


def _binary_search_i32(
  x: jax.Array,
  predicate_fn,
  lower_bound: int = 0,
  upper_bound: int = 2**30,
) -> jax.Array:
  """Binary search for i32 values with custom predicate.

  Finds the LARGEST threshold where predicate is FALSE.

  Args:
    x: Input array of shape [batch, vocab_size]
    predicate_fn: Function that takes (x, threshold) where threshold has shape
                  [batch, 1] and returns boolean array of shape [batch, 1]
    lower_bound: Lower bound for search (inclusive)
    upper_bound: Upper bound for search (inclusive)

  Returns:
    Threshold array of shape [batch]
  """
  batch_shape = x.shape[:-1]

  def loop_body(state):
    l, r = state
    # Integer midpoint (avoid overflow)
    pivot = (l // 2) + (r // 2) + ((l & 1) + (r & 1)) // 2

    # Evaluate predicate at midpoint
    predicate_true = predicate_fn(x, pivot)

    # Binary search: find largest value where predicate is FALSE
    # If predicate is TRUE at pivot, answer is < pivot, so update r = pivot - 1
    # If predicate is FALSE at pivot, answer might be pivot or > pivot, so update l = pivot
    l = jnp.where(predicate_true, l, pivot)
    r = jnp.where(predicate_true, pivot - 1, r)

    return (l, r)

  def cond(state):
    l, r = state
    # Continue while l < r
    return jnp.any(l < r)

  # Initialize bounds with shape [batch, 1]
  l = jnp.full(batch_shape + (1,), lower_bound, dtype=jnp.int32)
  r = jnp.full(batch_shape + (1,), upper_bound, dtype=jnp.int32)

  # Run binary search
  l, r = lax.while_loop(cond, loop_body, (l, r))

  # Return with shape [batch]
  return l.squeeze(-1)


def platform_portable_top_p(
  logits: jax.Array,
  top_p: float | jax.Array,
  scale: int = 2**30,
  replace_val: float = -1e12,
) -> jax.Array:
  """Platform-portable top-p sampling using high-precision arithmetic.

  All internal values are non-negative (probabilities scaled to [0, scale]).
  Uses 16-bit parts for safe summation of up to 2^15 values before harmonizing.

  Args:
    logits: Input logits of shape [batch, vocab_size]
    top_p: Probability threshold in range (0, 1]. Can be scalar or array of shape [batch]
    scale: Scale factor for converting f32 probs to i32 (default 2^30, must be < 2^31)
    replace_val: Value to use for masked elements

  Returns:
    Masked logits with same shape as input, where values outside top-p are replaced
  """
  # Convert top_p to array if scalar
  if isinstance(top_p, (float, int)):
    top_p_arr = jnp.full((logits.shape[0], 1), top_p, dtype=jnp.float32)
  else:
    top_p_arr = jnp.asarray(top_p, dtype=jnp.float32)
    if top_p_arr.ndim == 1:
      top_p_arr = top_p_arr[:, None]

  # 1. Compute unnormalized probabilities: exp(logits - max(logits))
  logits_max = logits.max(axis=1, keepdims=True)
  unnorm_probs_f32 = jnp.exp(logits - logits_max)

  # 2. Convert f32 probabilities to i32 range [0, scale]
  unnorm_probs_i32 = jnp.clip(
    (unnorm_probs_f32 * scale).astype(jnp.int32),
    0,
    scale
  )

  # 3. Convert to high-precision and sum
  unnorm_probs_hp = HighPrecisionUInt.from_i32_array(unnorm_probs_i32)
  total_sum_hp = unnorm_probs_hp.sum_dim1()

  # 4. Compute target sum: total_sum * top_p
  total_sum_f32 = total_sum_hp.to_f32()
  target_sum_f32 = total_sum_f32 * top_p_arr
  # Determine number of parts needed (u64 = 4 parts of 16 bits each)
  target_sum_hp = HighPrecisionUInt.from_f32(target_sum_f32, num_parts=len(total_sum_hp.parts))

  # 5. Binary search for threshold
  # Predicate: sum(unnorm_probs_i32 >= threshold) >= target_sum
  def predicate_fn(x, threshold):
    """Check if cumulative sum of values >= threshold exceeds target."""
    # x has shape [batch, vocab_size], threshold has shape [batch, 1]
    mask = x >= threshold
    masked_values = jnp.where(mask, x, 0)

    # Convert to high-precision and sum
    masked_hp = HighPrecisionUInt.from_i32_array(masked_values)
    cumsum_hp = masked_hp.sum_dim1()

    # Check if cumsum >= target_sum
    return cumsum_hp.compare_ge(target_sum_hp)

  threshold_i32 = _binary_search_i32(
    unnorm_probs_i32,
    predicate_fn,
    lower_bound=0,
    upper_bound=scale,
  )

  # 6. Apply mask to original logits
  threshold_i32_expanded = threshold_i32[:, None]
  mask = unnorm_probs_i32 >= threshold_i32_expanded

  return jnp.where(mask, logits, replace_val)
