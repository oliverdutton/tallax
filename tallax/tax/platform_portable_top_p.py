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


from tallax.tax.binary_search import binary_search

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
  # TODO Clip neccessary, try exhastive check?
  unnorm_probs_i32 = jnp.clip(
    (unnorm_probs_f32 * scale).astype(jnp.int32),
    0, scale
  )

  # 3. Convert to high-precision and sum
  unnorm_probs_hp = HighPrecisionUInt.from_i32_array(unnorm_probs_i32)
  total_sum_hp = unnorm_probs_hp.sum(axis=1)

  # 4. Compute target sum: total_sum * top_p
  total_sum_f32 = total_sum_hp.to_f32()
  target_sum_f32 = total_sum_f32 * top_p_arr
  # Determine number of parts needed (u64 = 4 parts of 16 bits each)
  target_sum_hp = HighPrecisionUInt.from_f32(target_sum_f32, num_parts=len(total_sum_hp.parts))

  # 5. Binary search for threshold
  # Predicate: sum(unnorm_probs_i32 >= threshold) < target_sum
  # Binary search finds the largest threshold where predicate is FALSE,
  # so we invert: return TRUE when cumsum < target_sum
  def predicate_fn(threshold):
    """Check if cumulative sum of values >= threshold is less than target."""
    # unnorm_probs_i32 has shape [batch, vocab_size]
    # threshold has shape [batch, 1] coming from binary_search broadcasting
    mask = unnorm_probs_i32 >= threshold
    masked_values = jnp.where(mask, unnorm_probs_i32, 0)

    # Convert to high-precision and sum
    masked_hp = HighPrecisionUInt.from_i32_array(masked_values)
    cumsum_hp = masked_hp.sum(axis=1)

    # Return TRUE when cumsum < target_sum (inverted for binary search)
    return ~cumsum_hp.compare_ge(target_sum_hp)

  bound_shape = (logits.shape[0], 1)
  threshold_i32, _ = binary_search(
    predicate_fn,
    *(jnp.full(bound_shape, v, jnp.int32) for v in (0, scale)),
  )
  # 6. Apply mask to original logits
  mask = unnorm_probs_i32 >= threshold_i32
  return jnp.where(mask, logits, replace_val)
