"""
Fused TPU sampling kernel implementing top-p filtering, temperature scaling,
and categorical sampling.
"""

from tallax.vllm.high_precision_uint import modulo_u128_u64
from tallax.tax.sparse_random import sparse_random_bits
import functools
import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec as P

from tallax.tax.bitonic.topk import bitonic_topk_arrays
from tallax.tax.cumsum import cumsum_arrays
from tallax.tax.gather import take_along_axis_arrays
from tallax.tax.utils import NUM_SUBLANES, NUM_LANES

_SAMPLING_EPS = 1e-5


def broadcast_to(x, shape):
  if x.shape[1] == shape[1] and shape[0] % NUM_LANES == 0 and x.shape[0] == 1:
    # workaround for jax issue #34001
    return pltpu.repeat(
      jnp.broadcast_to(x, (NUM_SUBLANES, shape[1])),
      shape[0] // NUM_SUBLANES,
      axis=0,
    )
  return jnp.broadcast_to(x, shape)


def top_p_mask(*, topk_logits, p, replace_val, axis):
  """
  Apply top-p filtering mask to sorted logits.

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

  # Compute softmax probabilities
  # For numerical stability, subtract max (pre-sorted so its the first element)
  exp_logits = jnp.exp(topk_logits - topk_logits[:1, :])
  probs = exp_logits / exp_logits.sum(axis=0, keepdims=True)

  # Top-p filtering using cumsum on sorted probabilities
  cumsum_probs = cumsum_arrays(probs, axis=0)

  # Find last idx where top-p probability mass is (over)covered
  threshold_idx = (cumsum_probs < p[None, :]).sum(0, keepdims=True)
  # Clamp for p=1.0 case
  threshold_idx = jnp.where(p[None, :] == 1.0, shape[0] - 1, threshold_idx)
  # vLLM current implementation uses binary search, computing a threshold.
  # so ties at the threshold are all included
  # we replicate that behavior here
  thresholds = take_along_axis_arrays(
    topk_logits, broadcast_to(threshold_idx, shape), axis=0
  )
  topp_logits = jnp.where(topk_logits >= thresholds, topk_logits, replace_val)

  return topp_logits


def top_p_integer_mask(*, topk_logits, p, axis):
  """
  Apply top-p filtering mask to sorted logits.

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

  # Compute softmax probabilities
  # For numerical stability, subtract max (pre-sorted so its the first element)
  exp_logits = jnp.exp(topk_logits - topk_logits[:1, :])
  scale = 2**24 - 1
  unnorm_probs_i32 = (exp_logits * scale).astype(jnp.int32)
  if unnorm_probs_i32.shape[1] > 2**7:
    raise NotImplementedError(
      "top_p_integer_mask only supports vocab_size <= 128, otherwise overflows i32 and higher precision simulation is required."
    )

  # Top-p filtering using cumsum on sorted probabilities
  cumsum_probs = cumsum_arrays(unnorm_probs_i32, axis=0)

  cumsum_threshold_i32 = (
    p[None, :] * unnorm_probs_i32.sum(0, keepdims=True).astype(jnp.float32)
  ).astype(jnp.int32)
  # Find last idx where top-p probability mass is (over)covered
  threshold_idx = (cumsum_probs < cumsum_threshold_i32).sum(0, keepdims=True)
  # Clamp for p=1.0 case
  threshold_idx = jnp.where(p[None, :] == 1.0, shape[0] - 1, threshold_idx)
  # vLLM current implementation uses binary search, computing a threshold.
  # so ties at the threshold are all included
  # we replicate that behavior here
  thresholds = take_along_axis_arrays(
    unnorm_probs_i32, broadcast_to(threshold_idx, shape), axis=0
  )
  return jnp.where(unnorm_probs_i32 >= thresholds, unnorm_probs_i32, 0)


def top_p_and_sample_arrays(
  *,
  topk_logits,
  topk_idx,
  random_u128_in_u32s,
  top_p,
  temperature,
  vocab_size,
  replace_val,
  dim0_offset: int = 0,
):
  """
  Implements top-p filtering, temperature scaling, and sampling.

  Args:
      topk_logits: Sorted logits of shape (batch_size, k)
      topk_idx: Indices corresponding to sorted logits of shape (batch_size, k)
      random_u128_in_u32s: Random u128 in u32s for sampling, list of four (batch_size, 1)
      top_p: Top-p threshold values, shape (batch_size,)
      temperature: Temperature values, shape (batch_size,)
      vocab_size: Vocabulary size for sampling
      replace_val: Value to replace filtered logits with
      dim0_offset: Offset for dim0 (batch) axis, used for sharding (default: 0)

  Returns:
      Sampled tokens of shape (batch_size,)
  """
  topk_logits = topk_logits.astype(jnp.float32)

  # To do reductions and broadcast across sublanes rather than lanes (which are slow)
  # we shift sampling to dim 0
  topk_logits = topk_logits.T
  topk_idx = topk_idx.T
  shape = topk_logits.shape

  topk_logits = topk_logits / temperature[None, :].astype(topk_logits.dtype)

  unnorm_probs_i32 = top_p_integer_mask(
    topk_logits=topk_logits, p=top_p, axis=0
  )

  # Reorder back to original order, to make sampling bitwise match with arbitrary k sampling kernel
  inverted_idxs, unnorm_probs_i32 = bitonic_topk_arrays(
    [-topk_idx, unnorm_probs_i32], k=topk_idx.shape[0], axis=0, num_keys=1
  )
  idxs = -inverted_idxs
  target_cumsum = modulo_u128_u64(
    random_u128_in_u32s,
    [
      jnp.zeros((shape[0], 1), dtype=jnp.uint32),
      unnorm_probs_i32.sum(0, keepdims=True).astype(jnp.uint32),
    ],
  )[1]
  # Within tile cumsum check
  # For high parallelism we make 128 (b, 1) tiles instead of several rounds of cumsum on (b, 128)
  iota1 = jax.lax.broadcasted_iota(jnp.int32, (shape[0], NUM_LANES), 0)
  cumsums = [
    (unnorm_probs_i32 * (iota1 <= i)).sum(0, keepdims=True)
    for i in range(NUM_LANES)
  ]
  threshold_local_idx = sum((c < target_cumsum) for c in cumsums)

  next_tokens = (idxs * (iota1 == threshold_local_idx)).sum(0)
  greedy_sampled = topk_idx[0, :]
  return jnp.where(temperature < _SAMPLING_EPS, greedy_sampled, next_tokens)


def top_p_and_sample_refs(
  topk_logits_ref,
  topk_idx_ref,
  rng_key_ref,
  top_p_ref,
  temperature_ref,
  dim0_offset_ref,
  sampled_tokens_ref,
  *,
  vocab_size: int,
  replace_val: float,
):
  """
  Fused kernel implementing top-p filtering, temperature scaling, and sampling.

  Args:
      topk_logits_ref: Reference to sorted logits
      topk_idx_ref: Reference to sorted indices
      rng_key_ref: Reference to RNG key (SMEM)
      top_p_ref: Reference to top-p values
      temperature_ref: Reference to temperature values
      dim0_offset_ref: Reference to dim0 offset for sharding (SMEM, shape (1,))
      sampled_tokens_ref: Reference to output sampled tokens
      vocab_size: Vocabulary size
      replace_val: Value to replace filtered logits with
  """
  sampled_tokens_ref[...] = top_p_and_sample_arrays(
    topk_logits=topk_logits_ref[...],
    topk_idx=topk_idx_ref[...],
    rng_key=rng_key_ref,  # SMEM, so keep as ref
    top_p=top_p_ref[...],
    temperature=temperature_ref[...],
    vocab_size=vocab_size,
    replace_val=replace_val,
    dim0_offset=dim0_offset_ref[0],  # Extract scalar from SMEM array
  )


def _top_p_and_sample(
  topk_logits: jax.Array,
  topk_idx: jax.Array,
  rng_key: jax.Array,  # threefry2x32 key
  top_p: jax.Array,
  temperature: jax.Array,
  *,
  vocab_size: int,
  replace_val: float,
  interpret: bool = False,
  dim0_offset: int = 0,
) -> jax.Array:
  """
  Fused TPU kernel for sampling with top-p filtering and temperature scaling.

  Args:
      topk_logits: Sorted logits of shape (batch_size, k)
      topk_idx: Indices corresponding to sorted logits of shape (batch_size, k)
      rng_key: RNG key for sampling, shape (2,)
      top_p: Top-p threshold values, scalar or shape (batch_size,)
      temperature: Temperature values, scalar or shape (batch_size,)
      vocab_size: Vocabulary size for sampling
      replace_val: Value to replace filtered logits with
      interpret: If True, run in CPU interpret mode (default: False)
      dim0_offset: Offset for dim0 (batch) axis, used for sharding (default: 0)
                   Must be computed outside pallas_call using lax.axis_index

  Returns:
      next_tokens: Sampled tokens of shape (batch_size,)
  """
  return pl.pallas_call(
    functools.partial(
      top_p_and_sample_refs,
      vocab_size=vocab_size,
      replace_val=replace_val,
    ),
    in_specs=(
      pl.BlockSpec(),
      pl.BlockSpec(),
      pl.BlockSpec(memory_space=pltpu.SMEM),
      pl.BlockSpec(),
      pl.BlockSpec(),
      pl.BlockSpec(memory_space=pltpu.SMEM),
    ),
    out_shape=jax.ShapeDtypeStruct(topk_logits.shape[:1], jnp.int32),
    interpret=interpret,
  )(
    topk_logits,
    topk_idx,
    rng_key.reshape(1, 2),
    top_p,
    temperature,
    jnp.array(dim0_offset, jnp.int32)[None],
  )


@functools.partial(
  jit,
  static_argnames=(
    "vocab_size",
    "replace_val",
    "interpret",
  ),
)
def top_p_and_sample(
  topk_logits: jax.Array,
  topk_idx: jax.Array,
  rng_key: jax.Array,
  top_p: jax.Array,
  temperature: jax.Array,
  *,
  vocab_size: int,
  replace_val: float,
  interpret: bool = False,
) -> jax.Array:
  """
  Sharded wrapper for top-p sampling with custom partitioning.

  Requires all axes except batch dim to be replicated. Batch dim can be sharded.

  Args:
      topk_logits: Sorted logits of shape (batch_size, k).
      topk_idx: Indices corresponding to sorted logits of shape (batch_size, k).
      rng_key: RNG key for sampling.
      top_p: Top-p threshold values.
      temperature: Temperature values.
      vocab_size: Total vocabulary size.
      replace_val: Value to replace filtered logits with.
      interpret: If True, run in CPU interpret mode (default: False).

  Returns:
      Sampled tokens of shape (batch_size,).
  """

  @custom_partitioning
  def sharded_top_p_and_sample(
    topk_logits, topk_idx, rng_key, top_p, temperature
  ):
    return _top_p_and_sample(
      topk_logits,
      topk_idx,
      rng_key,
      top_p,
      temperature,
      vocab_size=vocab_size,
      replace_val=replace_val,
      interpret=interpret,
    )

  def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
    # Output follows batch dimension of first input (replicated on other dims)
    batch_spec = arg_shapes[0].sharding.spec[0]
    return NamedSharding(mesh, P(batch_spec))

  def partition(mesh, arg_shapes, out_shapes):
    arg_shardings, out_shardings = jax.tree.map(
      lambda s: s.sharding, (arg_shapes, out_shapes)
    )
    batch_axis_name = arg_shardings[0].spec[0]

    def shmap_fn(topk_logits, topk_idx, rng_key, top_p, temperature):
      # Pass global sharded axis offset to maintain jax.random.categorical sampled values
      dim0_offset = 0
      if batch_axis_name is not None:
        dim0_offset = jax.lax.axis_index(batch_axis_name) * topk_logits.shape[0]
      return _top_p_and_sample(
        topk_logits,
        topk_idx,
        rng_key,
        top_p,
        temperature,
        vocab_size=vocab_size,
        replace_val=replace_val,
        interpret=interpret,
        dim0_offset=dim0_offset,
      )

    return mesh, shmap_fn, out_shardings, arg_shardings

  sharded_top_p_and_sample.def_partition(
    infer_sharding_from_operands=infer_sharding_from_operands,
    partition=partition,
    sharding_rule="b k, b k, r, b, b -> b",
    need_replication_factors=("k", "r"),
  )

  return sharded_top_p_and_sample(
    topk_logits, topk_idx, rng_key, top_p, temperature
  )
