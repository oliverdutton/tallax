"""Platform-portable top-p sampling using high-precision integer arithmetic.

This module implements a top-p (nucleus sampling) implementation that is:
1. Platform-portable: gives identical results across different hardware
2. Summation-order invariant: results don't depend on floating-point summation order
3. High-precision: uses safe integer arithmetic with dynamic bound tracking

The algorithm:
1. Convert logits to unnormalized probabilities: exp(logits - max(logits))
2. Scale f32 probabilities to i32 range [0, 2^24]
3. Sum using U48 (24-bit parts) and automatic normalization
4. Binary search for threshold where cumulative sum >= top_p * total_sum

Key optimization: Uses U48 with 24-bit parts and tracks max_value_bound
to minimize normalization. Only normalizes when max_value_bound >= 2^31 to prevent
i32 overflow.
"""

import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax.tax.utils import NUM_LANES, map_reduce
from tallax.vllm.high_precision_uint import U48, modulo_u128_u64
from tallax.vllm.binary_search import binary_search
from tallax.vllm.topk_mask import find_boundary_idx


def map_chunks(x, fn):
  """Apply a function to chunks of the input array to help force compiler fusion."""
  assert x.shape[1] % NUM_LANES == 0
  return jnp.concatenate(
    [fn(c) for c in jnp.split(x, x.shape[1] // NUM_LANES, 1)], axis=1
  )


def sample_probs(unnorm_probs_i32, random_u128_in_u32s, max_val=2**24 - 1):
  """Sample from unnormalized probabilities using high precision integers.

  Args:
    unnorm_probs_i32: Unnormalized probabilities in i32 format
    random_u128_in_u32s: Random u128 values in u32 format
    max_val: Maximum bound of unnorm_probs_i32 values

  Returns:
    Sampled token indices
  """
  total_sum_u48 = U48.map_reduce_sum(unnorm_probs_i32, max_val=max_val)
  sampled_u64_in_u32s = modulo_u128_u64(
    random_u128_in_u32s,
    total_sum_u48.to_u64_in_u32s(),
  )
  target_u48 = U48.from_u64_in_u32s(sampled_u64_in_u32s)
  return find_boundary_idx(
    unnorm_probs_i32,
    map_fn=lambda x: U48(x, max_val=max_val),
    target=target_u48,
  )


def topp_mask_ref_inputs(
  logits_ref,
  top_p_ref,
  *,
  scale_bits: int = 24,
  logits_max: jax.Array = None,
) -> jax.Array:
  """Core nucleus sampling logic for both standard and Pallas interfaces."""
  logits = logits_ref[...]
  try:
    top_p = top_p_ref[...]
  except (TypeError, IndexError):
    top_p = top_p_ref
  num_vals = logits.shape[1]
  # Convert top_p to array if scalar
  bound_shape = (logits.shape[0], NUM_LANES)
  top_p = jnp.broadcast_to(top_p, bound_shape)

  if logits_max is None:
    # 1. Compute unnormalized probabilities: exp(logits - max(logits)) and scale [0.,1.] to [0, 2^scale - 1]
    logits_max = map_reduce(
      logits, reduce_fn="max"
    )  # logits.max(axis=1, keepdims=True)

  scale = 2**scale_bits - 1
  unnorm_probs_i32 = map_chunks(
    logits,
    lambda logits: (jnp.exp(logits - logits_max) * scale).astype(jnp.int32),
  )

  # 2. Convert to U48 and sum safely using map_reduce_sum
  total_sum_u48 = U48.map_reduce_sum(unnorm_probs_i32, max_val=scale)

  # 3. Compute target sum: total_sum * top_p (also bounded by max_total_sum)
  target_sum_u48 = U48.from_f32(
    total_sum_u48.to_f32() * top_p, max_val=num_vals * scale
  )

  # 4. Binary search for threshold
  # Uses int32 during parallel reduction, then converts to U48
  def predicate_fn(threshold):
    """Check if cumulative sum of values >= threshold is less than target."""
    return (
      U48.map_reduce_sum(
        unnorm_probs_i32,
        max_val=scale,
        map_fn=lambda chunk: jnp.where(chunk >= threshold, chunk, 0),
      )
      < target_sum_u48
    )

  bound_shape = (logits.shape[0], NUM_LANES)
  threshold_i32, _, _ = binary_search(
    predicate_fn,
    *(jnp.full(bound_shape, v, jnp.int32) for v in (0, scale)),
    num_iter=scale_bits,
  )
  # 5. Apply mask to original logits
  # Broadcast threshold from (batch, NUM_LANES) to (batch, vocab_size)
  threshold_i32 = pltpu.repeat(
    threshold_i32, logits.shape[1] // NUM_LANES, axis=1
  )
  mask = unnorm_probs_i32 >= threshold_i32
  return jnp.where(mask, unnorm_probs_i32, 0)


def topp_mask(
  logits: jax.Array,
  top_p: jax.Array,
  scale_bits: int = 24,
  logits_max: jax.Array = None,
) -> jax.Array:
  """Platform-portable top-p sampling using high-precision arithmetic."""
  return topp_mask_ref_inputs(
    logits,
    top_p,
    scale_bits=scale_bits,
    logits_max=logits_max,
  )


def topp_mask_pallas_kernel(
  logits_ref,
  top_p_ref,
  output_ref,
  *,
  scale_bits: int,
):
  """Pallas kernel writing results to an output reference."""
  output_ref[...] = topp_mask_ref_inputs(
    logits_ref, top_p_ref, scale_bits=scale_bits
  )


@functools.partial(jax.jit, static_argnames=["scale_bits", "interpret"])
def topp_mask_pallas(
  logits: jax.Array,
  top_p: jax.Array,
  scale_bits: int = 24,
  interpret: bool = False,
) -> jax.Array:
  """Pallas-based interface for nucleus sampling.

  Args:
    logits: Input array of shape [batch, vocab_size]
    top_p: Number of top elements
    scale_bits: Precision for probability scaling
    interpret: Whether to use interpret mode

  Returns:
    Masked array
  """
  batch_size, _ = logits.shape
  top_p = jnp.broadcast_to(top_p, (batch_size, 1))
  output_shape = jax.ShapeDtypeStruct(logits.shape, logits.dtype)
  return pl.pallas_call(
    functools.partial(
      topp_mask_pallas_kernel,
      scale_bits=scale_bits,
    ),
    compiler_params=pltpu.CompilerParams(vmem_limit_bytes=int(0.9 * 2**27)),
    out_shape=output_shape,
    interpret=interpret,
  )(logits, top_p)
