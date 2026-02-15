"""Top-p (nucleus) masking via binary search over the full vocabulary.

Platform-portable top-p implementation that:
1. Converts logits to unnormalized i32 probabilities scaled to [0, 2^24]
2. Uses U48 safe summation to avoid i32 overflow
3. Binary searches for the i32 threshold where cumulative sum >= top_p * total_sum
4. Returns masked unnormalized probabilities (zeros for excluded tokens)

Never sorts the input.
"""

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from tallax.tax.utils import NUM_LANES, map_reduce
from tallax.vllm.utils.high_precision_uint import U48
from tallax.vllm.utils.binary_search import binary_search


def map_chunks(x, fn):
  """Apply a function to NUM_LANES-wide chunks to help force compiler fusion."""
  assert x.shape[1] % NUM_LANES == 0
  return jnp.concatenate(
    [fn(c) for c in jnp.split(x, x.shape[1] // NUM_LANES, 1)], axis=1
  )


def topp_mask(
  logits: jax.Array,
  top_p: jax.Array,
  *,
  scale_bits: int = 24,
  logits_max: jax.Array = None,
) -> jax.Array:
  """Apply top-p mask using binary search over integer probability space.

  Args:
    logits: Input logits [batch, vocab_size], float32
    top_p: Top-p threshold [batch, 1] or broadcastable
    scale_bits: Precision bits for probability scaling (default 24)
    logits_max: Pre-computed max logits [batch, 1] (optional, computed if None)

  Returns:
    Masked unnormalized probabilities in i32, zeros for excluded tokens.
    Shape [batch, vocab_size].
  """
  num_vals = logits.shape[1]
  bound_shape = (logits.shape[0], NUM_LANES)
  top_p = jnp.broadcast_to(top_p, bound_shape)

  if logits_max is None:
    logits_max = map_reduce(logits, reduce_fn="max")

  scale = 2**scale_bits - 1
  unnorm_probs_i32 = map_chunks(
    logits,
    lambda logits: (jnp.exp(logits - logits_max) * scale).astype(jnp.int32),
  )

  # Safe sum using U48 to avoid i32 overflow
  total_sum_u48 = U48.map_reduce_sum(unnorm_probs_i32, max_val=scale)

  # Target sum: total_sum * top_p
  target_sum_u48 = U48.from_f32(
    total_sum_u48.to_f32() * top_p, max_val=num_vals * scale
  )

  # Binary search for threshold in i32 space
  def predicate_fn(threshold):
    return (
      U48.map_reduce_sum(
        unnorm_probs_i32,
        max_val=scale,
        map_fn=lambda chunk: jnp.where(chunk >= threshold, chunk, 0),
      )
      < target_sum_u48
    )

  threshold_i32, _, _ = binary_search(
    predicate_fn,
    *(jnp.full(bound_shape, v, jnp.int32) for v in (0, scale)),
    num_iter=scale_bits,
  )

  # Apply mask
  threshold_i32 = pltpu.repeat(
    threshold_i32, logits.shape[1] // NUM_LANES, axis=1
  )
  mask = unnorm_probs_i32 >= threshold_i32
  return jnp.where(mask, unnorm_probs_i32, 0)
