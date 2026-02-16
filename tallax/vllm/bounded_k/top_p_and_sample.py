"""Bounded-k top-p filtering and sampling using bitonic sort.

This kernel assumes logits have already been reduced to top-k elements
(via bitonic sort). It operates on the sorted subset:
  1. Temperature scaling
  2. top_p_integer_mask: cumsum on sorted i32 probs, threshold by cumulative sum
  3. Re-sort back to original index order via bitonic_topk_arrays
  4. Sample using modulo_u128_u64 on cumsum of unnorm probs

Only supports vocab_size (k) <= 128 due to i32 overflow constraints.
"""

import functools
import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec as P

from tallax.vllm.utils.high_precision_uint import modulo_u128_u64
from tallax.vllm.utils.high_precision_uint import sample_random_u128_in_u32s
from tallax.tax.bitonic.topk import bitonic_topk_arrays
from tallax.tax.cumsum import cumsum_arrays
from tallax.tax.gather import take_along_axis_arrays
from tallax.tax.utils import NUM_SUBLANES, NUM_LANES

_SAMPLING_EPS = 1e-5


def broadcast_to(x, shape):
  if x.shape[1] == shape[1] and shape[0] % NUM_LANES == 0 and x.shape[0] == 1:
    return pltpu.repeat(
      jnp.broadcast_to(x, (NUM_SUBLANES, shape[1])),
      shape[0] // NUM_SUBLANES,
      axis=0,
    )
  return jnp.broadcast_to(x, shape)


def top_p_integer_mask(*, topk_logits, p, axis):
  """Apply top-p filtering on sorted logits using integer arithmetic.

  Converts softmax probabilities to i32 scaled values, computes cumsum,
  and masks tokens below the top-p threshold. Only supports k <= 128.

  Args:
    topk_logits: Sorted logits (descending order) along axis
    p: Top-p threshold(s)
    axis: Axis along which to apply filtering (must be 0)

  Returns:
    Masked unnormalized i32 probabilities (zeros for excluded tokens)
  """
  if axis != 0:
    raise NotImplementedError("top_p_integer_mask only supports axis=0")

  shape = topk_logits.shape

  exp_logits = jnp.exp(topk_logits - topk_logits[:1, :])
  scale = 2**24 - 1
  unnorm_probs_i32 = (exp_logits * scale).astype(jnp.int32)
  if unnorm_probs_i32.shape[axis] > 2**7:
    raise NotImplementedError(
      "top_p_integer_mask only supports vocab_size <= 128, otherwise overflows i32."
    )

  cumsum_probs = cumsum_arrays(unnorm_probs_i32, axis=0)

  cumsum_threshold_i32 = (
    p[None, :] * unnorm_probs_i32.sum(0, keepdims=True).astype(jnp.float32)
  ).astype(jnp.int32)
  threshold_idx = (cumsum_probs < cumsum_threshold_i32).sum(0, keepdims=True)
  threshold_idx = jnp.where(p[None, :] == 1.0, shape[0] - 1, threshold_idx)
  thresholds = take_along_axis_arrays(
    unnorm_probs_i32, broadcast_to(threshold_idx, shape), axis=0
  )
  return jnp.where(unnorm_probs_i32 >= thresholds, unnorm_probs_i32, 0)


def top_p_mask(*, topk_logits, p, replace_val, axis):
  """Apply top-p filtering mask to sorted logits (float version for backwards compat).

  Args:
    topk_logits: Sorted logits (descending order)
    p: Top-p threshold(s)
    replace_val: Value to replace filtered logits with
    axis: Axis along which to apply filtering (must be 0)

  Returns:
    Masked logits with values outside top-p set to replace_val
  """
  if axis != 0:
    raise NotImplementedError("topp_mask only supports axis=0")

  shape = topk_logits.shape
  exp_logits = jnp.exp(topk_logits - topk_logits[:1, :])
  probs = exp_logits / exp_logits.sum(axis=0, keepdims=True)
  cumsum_probs = cumsum_arrays(probs, axis=0)

  threshold_idx = (cumsum_probs < p[None, :]).sum(0, keepdims=True)
  threshold_idx = jnp.where(p[None, :] == 1.0, shape[0] - 1, threshold_idx)
  thresholds = take_along_axis_arrays(
    topk_logits, broadcast_to(threshold_idx, shape), axis=0
  )
  return jnp.where(topk_logits >= thresholds, topk_logits, replace_val)


def top_p_and_sample_arrays(
  *,
  topk_logits,
  topk_idx,
  random_u128_in_u32s,
  top_p,
  temperature,
  debug=False,
):
  """Fused top-p filtering + sampling on pre-sorted top-k logits.

  Args:
    topk_logits: Sorted logits of shape (batch_size, k)
    topk_idx: Indices corresponding to sorted logits (batch_size, k)
    random_u128_in_u32s: List of 4 u32 arrays for random sampling
    top_p: Top-p threshold values, shape (batch_size,)
    temperature: Temperature values, shape (batch_size,)
    debug: If True, return (tokens, debug_results) with intermediate values

  Returns:
    Sampled tokens of shape (batch_size,), or (tokens, debug_dict) if debug=True
  """
  topk_logits = topk_logits.astype(jnp.float32)

  # Store original shape for debug
  batch_size, k = topk_logits.shape

  # Shift to dim 0 for sublane-based reductions
  topk_logits_transposed = topk_logits.T
  topk_idx_transposed = topk_idx.T
  random_u128_in_u32s_transposed = [x.T for x in random_u128_in_u32s]
  shape = topk_logits_transposed.shape

  # Greedy sample (before temperature scaling)
  greedy_sampled = topk_idx[:, :1]

  # Temperature scaling
  topk_logits_scaled = topk_logits_transposed / temperature[None, :].astype(
    topk_logits_transposed.dtype
  )

  # Top-p masking in i32 space
  unnorm_probs_i32_sorted = top_p_integer_mask(
    topk_logits=topk_logits_scaled, p=top_p, axis=0
  )

  # Re-sort back to original index order for bitwise-matching sampling
  inverted_idxs, unnorm_probs_i32_unsorted = bitonic_topk_arrays(
    [-topk_idx_transposed, unnorm_probs_i32_sorted],
    k=topk_idx_transposed.shape[0],
    axis=0,
    num_keys=1,
  )
  idxs = -inverted_idxs

  # Sample from the unnormalized probabilities
  target_cumsum = modulo_u128_u64(
    random_u128_in_u32s_transposed,
    [
      jnp.zeros((1, shape[1]), dtype=jnp.uint32),
      unnorm_probs_i32_unsorted.sum(0, keepdims=True).astype(jnp.uint32),
    ],
  )[1]
  cumsum = cumsum_arrays(unnorm_probs_i32_unsorted, axis=0)
  threshold_local_idx = (cumsum < target_cumsum).sum(0, keepdims=True)

  next_tokens = (
    idxs
    * (
      jax.lax.broadcasted_iota(jnp.int32, idxs.shape, 0) == threshold_local_idx
    )
  ).sum(0, keepdims=True)
  result = jnp.where(
    temperature[None, :] < _SAMPLING_EPS, greedy_sampled, next_tokens
  )

  if not debug:
    return result

  # Build debug output matching the reference format but for the k-slice
  random_unnorm_cdf_sampled_low = target_cumsum.T

  debug_results = {
    "greedy_sampled": greedy_sampled,
    "topk_logits_unsorted": topk_logits.T,  # Original sorted logits in batch-first format
    "topk_topp_unnorm_probs_i32_unsorted": unnorm_probs_i32_unsorted.T,  # Transpose back to [batch, k]
    "random_unnorm_cdf_sampled": (
      jnp.zeros_like(random_unnorm_cdf_sampled_low),
      random_unnorm_cdf_sampled_low,
    ),
    "next_tokens": next_tokens,
  }

  return result, debug_results


def top_p_and_sample_refs(
  topk_logits_ref,
  topk_idx_ref,
  random_u128_in_u32s_refs,
  top_p_ref,
  temperature_ref,
  sampled_tokens_ref,
  debug_arrays_ref=None,
):
  """Pallas kernel body for top-p filtering + sampling."""
  result = top_p_and_sample_arrays(
    topk_logits=topk_logits_ref[...],
    topk_idx=topk_idx_ref[...],
    random_u128_in_u32s=[ref[...] for ref in random_u128_in_u32s_refs],
    top_p=top_p_ref[...],
    temperature=temperature_ref[...],
    debug=debug_arrays_ref is not None,
  )
  if debug_arrays_ref is None:
    sampled_tokens_ref[...] = result
    return

  sampled_tokens, debug_results = result
  sampled_tokens_ref[...] = sampled_tokens
  for key, val in debug_results.items():
    if isinstance(val, tuple):
      for ref, v in zip(debug_arrays_ref[key], val, strict=True):
        ref[...] = v
    else:
      debug_arrays_ref[key][...] = val


def _top_p_and_sample(
  topk_logits, topk_idx, random_u128_in_u32s, top_p, temperature, *, debug=False
):
  batch_size, k = topk_logits.shape
  top_p, temperature = (
    jnp.broadcast_to(v, (batch_size,)) for v in (top_p, temperature)
  )

  output_shape = jax.ShapeDtypeStruct((1, batch_size), jnp.int32)

  if debug:
    debug_out_shapes = {
      "greedy_sampled": jax.ShapeDtypeStruct((batch_size, 1), jnp.int32),
      "topk_logits_unsorted": jax.ShapeDtypeStruct(
        (batch_size, k), jnp.float32
      ),
      "topk_topp_unnorm_probs_i32_unsorted": jax.ShapeDtypeStruct(
        (batch_size, k), jnp.int32
      ),
      "random_unnorm_cdf_sampled": (
        jax.ShapeDtypeStruct((batch_size, 1), jnp.uint32),
      )
      * 2,
      "next_tokens": jax.ShapeDtypeStruct((batch_size, 1), jnp.int32),
    }
  else:
    debug_out_shapes = None

  sampled_tokens, debug_aux = pl.pallas_call(
    top_p_and_sample_refs,
    out_shape=(output_shape, debug_out_shapes),
  )(
    topk_logits,
    topk_idx,
    random_u128_in_u32s,
    top_p,
    temperature,
  )

  sampled_tokens = sampled_tokens.squeeze(0)
  if not debug:
    return sampled_tokens

  debug_aux = jax.tree.map(
    lambda x: x.squeeze(1) if x.shape[1] == 1 else x, debug_aux
  )
  return sampled_tokens, debug_aux


@functools.partial(
  jit,
  static_argnames=("debug",),
)
def topp_and_sample(
  topk_logits: jax.Array,
  topk_idx: jax.Array,
  rng_key: jax.Array,
  top_p: jax.Array,
  temperature: jax.Array,
  *,
  debug: bool = False,
) -> jax.Array:
  """Sharded wrapper for top-p sampling with custom partitioning.

  Requires all axes except batch dim to be replicated. Batch dim can be sharded.

  Args:
    topk_logits: Sorted logits of shape (batch_size, k).
    topk_idx: Indices corresponding to sorted logits (batch_size, k).
    rng_key: RNG key for sampling.
    top_p: Top-p threshold values.
    temperature: Temperature values.
    vocab_size: Total vocabulary size.
    max_k: Maximum k value (bounded implementation supports k <= 128).
    replace_val: Value to replace filtered logits with.
    interpret: If True, run in CPU interpret mode.
    debug: If True, return (tokens, debug_results) with intermediate values.

  Returns:
    Sampled tokens of shape (batch_size,), or (tokens, debug_dict) if debug=True.
  """
  # Generate random u128 outside the sharded function
  batch_size = topk_logits.shape[0]
  random_u128_in_u32s = list(
    sample_random_u128_in_u32s(rng_key, (batch_size, 1))
  )

  out_shapes = jax.eval_shape(
    functools.partial(
      _top_p_and_sample,
      debug=debug,
    ),
    topk_logits,
    topk_idx,
    random_u128_in_u32s,
    top_p,
    temperature,
  )

  @custom_partitioning
  def sharded_top_p_and_sample(
    topk_logits, topk_idx, random_u128_in_u32s, top_p, temperature
  ):
    return jax.tree.leaves(
      _top_p_and_sample(
        topk_logits,
        topk_idx,
        random_u128_in_u32s,
        top_p,
        temperature,
        debug=debug,
      )
    )

  def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
    batch_spec = arg_shapes[0].sharding.spec[0]
    return [
      NamedSharding(mesh, P(batch_spec))
      for _ in range(len(jax.tree.leaves(out_shapes)))
    ]

  def partition(mesh, arg_shapes, out_shapes):
    arg_shardings, out_shardings = jax.tree.map(
      lambda s: s.sharding, (arg_shapes, out_shapes)
    )

    def shmap_fn(
      topk_logits, topk_idx, random_u128_in_u32s, top_p, temperature
    ):
      return jax.tree.leaves(
        _top_p_and_sample(
          topk_logits,
          topk_idx,
          random_u128_in_u32s,
          top_p,
          temperature,
          debug=debug,
        )
      )

    return mesh, shmap_fn, out_shardings, arg_shardings

  sharding_rule = "b k, b k, r r r r, b, b -> b"
  if debug:
    sharding_rule += ", b" * (len(jax.tree.leaves(out_shapes)) - 1)
  sharded_top_p_and_sample.def_partition(
    infer_sharding_from_operands=infer_sharding_from_operands,
    partition=partition,
    sharding_rule=sharding_rule,
    need_replication_factors=("k", "r"),
  )
  flat_outs = sharded_top_p_and_sample(
    topk_logits, topk_idx, random_u128_in_u32s, top_p, temperature
  )
  sampled_tokens, debug_aux = jax.tree.unflatten(
    jax.tree.structure(out_shapes), flat_outs
  )
  if debug:
    return sampled_tokens, debug_aux
  return sampled_tokens
