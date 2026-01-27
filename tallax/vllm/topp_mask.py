"""Platform-portable top-p sampling using high-precision integer arithmetic.

This module implements a top-p (nucleus sampling) implementation that is:
1. Platform-portable: gives identical results across different hardware
2. Summation-order invariant: results don't depend on floating-point summation order
3. High-precision: uses safe integer arithmetic with dynamic bound tracking

The algorithm:
1. Convert logits to unnormalized probabilities: exp(logits - max(logits))
2. Scale f32 probabilities to i32 range [0, 2^24]
3. Sum using U48 (48-bit with 24-bit parts) and automatic harmonization
4. Binary search for threshold where cumulative sum >= top_p * total_sum

Key optimization: Uses U48 with 24-bit parts and tracks max_value_bound to minimize harmonization.
Only harmonizes when max_value_bound >= 2^31 to prevent i32 overflow.
"""

import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax.tax.utils import NUM_LANES
from tallax.vllm.high_precision_uint import U48

from tallax.vllm.binary_search import binary_search


@functools.partial(jax.jit, static_argnames=('compare_fn', 'num_parallel', 'apply_post_partial_sums_fn'))
def reduce_compare_sum(vals, threshold, compare_fn, num_parallel: int=7, apply_post_partial_sums_fn=None):
  """Computes (vals > threshold).sum(axis=1, keepdims=True) efficiently.

  Works with int32 arrays. After parallel reduction, optionally converts to U48
  via apply_post_partial_sums_fn for the tree reduction phase.
  """
  vals_shape_1 = vals.shape[1]

  # Fast path for small inputs
  if num_parallel == 0 or vals_shape_1 < NUM_LANES:
    assert threshold.shape[1] == 1
    return compare_fn(vals, threshold).sum(1, keepdims=True)

  # Chunk and compute partial sums
  num_chunks = vals_shape_1 // NUM_LANES
  chunks = [vals[:, i*NUM_LANES:(i+1)*NUM_LANES] for i in range(num_chunks)]
  num_parallel = min(num_parallel, len(chunks))

  partial_sums = [compare_fn(chunk, threshold).astype(jnp.int32) for chunk in chunks[:num_parallel]]
  for i, chunk in enumerate(chunks[num_parallel:]):
    partial_sums[i % num_parallel] += compare_fn(chunk, threshold)

  # Convert to HighPrecisionUInt if requested (for tree reduction with bound tracking)
  if apply_post_partial_sums_fn is not None:
    partial_sums = list(map(apply_post_partial_sums_fn, partial_sums))

  # Tree reduction (works with both int32 and HighPrecisionUInt via duck typing)
  while len(partial_sums) > 1:
    n = len(partial_sums)
    for i in range(n // 2):
      partial_sums[i] = partial_sums[i] + partial_sums[i + pl.cdiv(n, 2)]
    partial_sums = partial_sums[:pl.cdiv(n, 2)]

  # Sum final dimension
  full_sum = partial_sums[0].sum(1, keepdims=True)

  # Handle remainder chunk
  has_remainder = vals_shape_1 % NUM_LANES
  if has_remainder:
    remainder_chunk_sum = compare_fn(vals[:, num_chunks * NUM_LANES:], threshold[:,:1]).sum(1, keepdims=True)
    full_sum = full_sum + remainder_chunk_sum

  return full_sum


def topp_mask(
  logits: jax.Array,
  top_p: jax.Array,
  scale_bits: int = 24,
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
  num_vals = logits.shape[1]
  # Convert top_p to array if scalar
  top_p = jnp.broadcast_to(top_p, (logits.shape[0], 1))

  # 1. Compute unnormalized probabilities: exp(logits - max(logits))
  logits_max = logits.max(axis=1, keepdims=True)
  unnorm_probs_f32 = jnp.exp(logits - logits_max)

  # 2. Convert f32 probabilities to i32 range [0, scale]
  scale = 2**scale_bits - 1
  unnorm_probs_i32 = (unnorm_probs_f32 * scale).astype(jnp.int32)

  safe_reduce_size = 2 ** (31 - scale_bits)

  # 3. Convert to U48 and sum
  unnorm_probs_u48 = U48.from_i32_array(unnorm_probs_i32, max_val=scale)
  total_sum_u48 = unnorm_probs_u48.sum(axis=1, keepdims=True)

  bound_shape = (logits.shape[0], NUM_LANES)

  # 4. Compute target sum: total_sum * top_p (also bounded by max_total_sum)
  target_sum_u48 = U48.from_f32(
    total_sum_u48.to_f32() * top_p,
    max_val=num_vals * scale)

  # Calculate bounds for partial sums after parallel reduction
  # Each partial_sum accumulates ~safe_reduce_size values
  partial_sum_max = scale * safe_reduce_size

  # 5. Binary search for threshold
  # Uses int32 during parallel reduction, then converts to U48
  def predicate_fn(threshold):
    """Check if cumulative sum of values >= threshold is less than target."""
    compare_fn = lambda lhs, rhs: jnp.where(lhs >= rhs, lhs, 0)
    return reduce_compare_sum(
      unnorm_probs_i32,
      threshold,
      compare_fn,
      num_parallel=pl.cdiv(num_vals, safe_reduce_size),
      apply_post_partial_sums_fn=lambda x: U48.from_i32_array(x, max_val=partial_sum_max),
    ) < target_sum_u48

  bound_shape = (logits.shape[0], NUM_LANES)
  threshold_i32, _, _ = binary_search(
    predicate_fn,
    *(jnp.full(bound_shape, v, jnp.int32) for v in (0, scale)),
    num_iter=scale_bits,
  )
  # 6. Apply mask to original logits
  # Broadcast threshold from (batch, NUM_LANES) to (batch, vocab_size)
  threshold_i32 = jnp.tile(threshold_i32, (1, logits.shape[1] // NUM_LANES))
  mask = unnorm_probs_i32 >= threshold_i32
  return jnp.where(mask, logits, replace_val)
