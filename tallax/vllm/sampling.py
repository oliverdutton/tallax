"""
Fused TPU sampling kernel implementing top-p filtering, temperature scaling,
and categorical sampling in a single Pallas kernel.
"""

import jax
import jax.numpy as jnp
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec as P

from tallax.tax.bitonic.topk import bitonic_topk_arrays as bitonic_topk
from tallax.vllm.top_p_and_sample import top_p_and_sample
from tallax.tax.divide_and_filter_topk.topk import top_bounded_k
from tallax.tax.utils import NUM_LANES


def _topk_with_sharding(
  logits: jax.Array,
  k: jax.Array,
  replace_val: float | int,
  max_k: int,
  num_bins,
  bins_topm_schedule: tuple[int, ...],
):
  def _topk(logits: jax.Array, k: jax.Array):
    return top_bounded_k(
      logits,
      k=k,
      max_k=max_k,
      guarantee_convergence=True,
      num_bins=num_bins,
      bins_topm_schedule=bins_topm_schedule,
      replace_val=replace_val,
    )

  @custom_partitioning
  def sharded_topk(logits, k):
    return _topk(logits, k)

  def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
    logits_spec = arg_shapes[0].sharding.spec
    return (NamedSharding(mesh, P(logits_spec[0], None)),) * 2

  def partition(mesh, arg_shapes, out_shapes):
    arg_shardings, out_shardings = jax.tree.map(
      lambda s: s.sharding, (arg_shapes, out_shapes)
    )
    axis_name = arg_shardings[0].spec[1]

    def shmap_fn(logits, k):
      topk_logits, topk_idxs = _topk(logits, k)
      if axis_name is None:
        return topk_logits, topk_idxs
      # convert idxs to global frame
      i = jax.lax.axis_index(axis_name)
      topk_idxs += i * logits.shape[1]
      # all-gather and top-k
      operands = [
        jax.lax.all_gather(x, axis_name, axis=1)
        for x in (topk_logits, topk_idxs)
      ]
      topk_logits, topk_idxs = bitonic_topk(operands, k=max_k)
      topk_logits = jnp.where(
        jax.lax.broadcasted_iota(jnp.int32, topk_logits.shape, 1) < k[:, None],
        topk_logits,
        replace_val,
      )
      return topk_logits, topk_idxs

    return mesh, shmap_fn, out_shardings, arg_shardings

  sharded_topk.def_partition(
    infer_sharding_from_operands=infer_sharding_from_operands,
    partition=partition,
    sharding_rule="b v, b -> b k, b k",
  )
  return sharded_topk(logits, k)


def topk_topp_and_sample(
  rng_key,
  logits,
  tpu_sampling_metadata,
  max_k=NUM_LANES,
  num_bins=256,
  bins_topm_schedule=(5, 9),
):
  vocab_size = logits.shape[1]
  topk_logits, topk_idxs = _topk_with_sharding(
    logits,
    k=tpu_sampling_metadata.top_k,
    replace_val=-1e12,
    max_k=max_k,
    num_bins=num_bins,
    bins_topm_schedule=bins_topm_schedule,
  )
  return top_p_and_sample(
    topk_logits,
    topk_idxs,
    rng_key,
    top_p=tpu_sampling_metadata.top_p,
    temperature=tpu_sampling_metadata.temperature,
    vocab_size=vocab_size,
    replace_val=-1e12,
  )
