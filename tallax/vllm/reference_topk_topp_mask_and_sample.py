import numpy as np
import jax
import jax.numpy as jnp

from tallax.vllm.topk_topp_mask_and_sample import (
  sample_random_u128_in_u32s,
  _SAMPLING_EPS,
)


def _python_u128_modulo_u64_impl(r0, r1, r2, r3, m0, m1):
  """Python implementation of 128-bit % 64-bit using arbitrary precision integers."""
  r0, r1, r2, r3 = [np.array(x, dtype=object) for x in (r0, r1, r2, r3)]
  m0, m1 = [np.array(x, dtype=object) for x in (m0, m1)]

  # Reconstruct 128-bit random value (Big Endian parts)
  val_128 = (r0 << 96) | (r1 << 64) | (r2 << 32) | r3

  # Reconstruct 64-bit modulus (Big Endian parts)
  val_mod = (m0 << 32) | m1

  # Avoid division by zero (should not happen with valid probability sums)
  # But handle it gracefully if val_mod is 0 (empty vocab or 0 probs?)
  # In simulation, we might encounter 0 sum if all masked.
  # U48 arithmetic handles it implicitly or crashes? python % 0 raises.
  # Let's return 0 if mod is 0.

  res = np.where(
    val_mod == 0, np.uint64(0), (val_128 % val_mod).astype(np.uint64)
  )
  scale = 2**32
  return tuple(x.astype(np.uint32) for x in [res // scale, res % scale])


def u128_modulo_u64_pure_callback(r_parts, m_parts):
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
  block_token: int = 8,
  interpret: bool = False,
  debug: bool = False,
) -> jax.Array:
  """Reference implementation in pure JAX of topk_topp_mask_and_sample."""
  assert stable
  with jax.enable_x64(True):
    shape = logits.shape

    logits = logits.astype(jnp.float32)
    greedy_sampled = logits.argmax(axis=1)

    sorted_indices = logits.argsort(axis=1, stable=True, descending=True)
    sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=1)

    # Top-k
    sorted_logits = jnp.where(
      jnp.arange(shape[1])[None, :] < k[:, None], sorted_logits, replace_val
    )

    # Temperature
    sorted_logits /= temperature[:, None]

    # Top-p, convert probs to int values [0,2**24)
    scale = 2**24 - 1
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

    # revert back to original order for sampling in int space
    total_sum = unnorm_probs_i32.sum(axis=1, keepdims=True)
    # For random sampling, we sample a random u128 value, then modulo the maxval
    random_u128_in_u32s = sample_random_u128_in_u32s(rng_key, (shape[0], 1))
    sampled_total = u128_modulo_u64_pure_callback(
      random_u128_in_u32s,
      # Total sum of i64 in u32s format
      [
        (total_sum // 2**32).astype(jnp.uint32),
        (total_sum % 2**32).astype(jnp.uint32),
      ],
    )
    next_tokens = (unnorm_probs_i32.cumsum(axis=1) < sampled_total).sum(axis=1)
    return jnp.where(temperature < _SAMPLING_EPS, greedy_sampled, next_tokens)
