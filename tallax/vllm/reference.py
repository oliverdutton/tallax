"""Pure JAX reference implementation for top-k + top-p + sampling.

No Pallas, no TPU-specific operations. Uses jax.enable_x64 for exact
i64 arithmetic and jax.pure_callback for arbitrary-precision u128 % u64.

This is the ground truth that both the arbitrary_k and bounded_k kernels
should match (modulo the intermediate representation).
"""

from collections import OrderedDict
import numpy as np
import jax
import jax.numpy as jnp

from tallax.vllm.utils.high_precision_uint import sample_random_u128_in_u32s

_SAMPLING_EPS = 1e-5
_SCALE_BITS = 24


def _python_u128_modulo_u64_impl(r0, r1, r2, r3, m0, m1):
  """Python implementation of 128-bit % 64-bit using arbitrary precision integers."""
  r0, r1, r2, r3 = [np.array(x, dtype=object) for x in (r0, r1, r2, r3)]
  m0, m1 = [np.array(x, dtype=object) for x in (m0, m1)]

  val_128 = (r0 << 96) | (r1 << 64) | (r2 << 32) | r3
  val_mod = (m0 << 32) | m1

  res = np.where(
    val_mod == 0, np.uint64(0), (val_128 % val_mod).astype(np.uint64)
  )
  scale = 2**32
  return tuple(x.astype(np.uint32) for x in [res // scale, res % scale])


def u128_modulo_u64_pure_callback(r_parts, m_parts):
  """Compute u128 % u64 via pure_callback to Python arbitrary precision."""
  high, low = jax.pure_callback(
    _python_u128_modulo_u64_impl,
    (jax.ShapeDtypeStruct(r_parts[0].shape, jnp.uint32),) * 2,
    *r_parts,
    *m_parts,
  )
  return (high.astype(jnp.int64) << 32) + low.astype(jnp.int64)


def reference_topk_topp_mask_and_sample(
  logits: jax.Array,
  rng_key: jax.Array,
  k: jax.Array,
  p: jax.Array,
  temperature: jax.Array,
  *,
  stable: bool = True,
  replace_val: float = -1e12,
  debug: bool = False,
) -> jax.Array:
  """Reference implementation of topk + topp + sample in pure JAX.

  Uses x64 arithmetic for exact computation. No Pallas.

  Args:
    logits: Input logits [batch, vocab_size], any float dtype
    rng_key: JAX RNG key
    k: Top-k values [batch] or scalar
    p: Top-p values [batch] or scalar
    temperature: Temperature values [batch] or scalar
    stable: Must be True (reference assumes stable)
    replace_val: Replacement value for masked elements
    debug: If True, return (tokens, debug_results) with intermediate values

  Returns:
    Sampled token indices [batch], or (tokens, debug_dict) if debug=True
  """
  assert stable
  with jax.enable_x64(True):
    shape = logits.shape
    scale = 2**_SCALE_BITS - 1

    logits = logits.astype(jnp.float32)

    # Stage 1: Greedy argmax
    greedy_sampled = logits.argmax(axis=1)

    # Stage 2: Top-k via sort
    sorted_indices = logits.argsort(axis=1, stable=True, descending=True)
    sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=1)
    sorted_logits = jnp.where(
      jnp.arange(shape[1])[None, :] < k[:, None], sorted_logits, replace_val
    )
    topk_logits_unsorted = jnp.take_along_axis(
      sorted_logits, sorted_indices.argsort(), axis=1
    )

    # Stage 3: Temperature
    sorted_logits /= temperature[:, None]

    # Stage 4: Top-p in i32 space
    sorted_unnorm_probs_i32 = (
      jnp.exp(sorted_logits - sorted_logits[:, :1]) * scale
    ).astype(jnp.int64)
    top_p_threshold_idx = (
      sorted_unnorm_probs_i32.cumsum(axis=1)
      < (
        sorted_unnorm_probs_i32.sum(axis=1, keepdims=True) * p[:, None]
      ).astype(jnp.int64)
    ).sum(axis=1, keepdims=True)
    threshold = jnp.take_along_axis(
      sorted_unnorm_probs_i32, top_p_threshold_idx, axis=1
    )
    sorted_unnorm_probs_i32 = jnp.where(
      sorted_unnorm_probs_i32 < threshold, 0, sorted_unnorm_probs_i32
    )

    # Reverse back to original order
    unnorm_probs_i32 = jnp.take_along_axis(
      sorted_unnorm_probs_i32, sorted_indices.argsort(), axis=1
    )

    # Stage 5: Sample in integer space
    total_sum = unnorm_probs_i32.sum(axis=1, keepdims=True)
    random_u128_in_u32s = sample_random_u128_in_u32s(rng_key, (shape[0], 1))
    sampled_total = u128_modulo_u64_pure_callback(
      random_u128_in_u32s,
      [
        (total_sum // 2**32).astype(jnp.uint32),
        (total_sum % 2**32).astype(jnp.uint32),
      ],
    )
    next_tokens = (unnorm_probs_i32.cumsum(axis=1) < sampled_total).sum(axis=1)
    result = jnp.where(temperature < _SAMPLING_EPS, greedy_sampled, next_tokens)

    if not debug:
      return result

    debug_results = OrderedDict([
      ("greedy_sampled", greedy_sampled),
      ("topk_logits_unsorted", topk_logits_unsorted),
      ("topk_topp_unnorm_probs_i32_unsorted", unnorm_probs_i32),
      ("random_unnorm_cdf_sampled", sampled_total),
      ("next_tokens", next_tokens),
    ])
    return result, debug_results
