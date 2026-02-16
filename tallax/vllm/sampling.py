"""vLLM top-k top-p sampling, using two stages.

Stage 1 (bounded_k path): Divide-and-filter top-k to reduce vocab to k elements.
Stage 2 (bounded_k path): Top-p + sample on the reduced sorted subset.

For direct full-vocab binary-search path, use arbitrary_k.topk_topp_mask_and_sample.
"""

import functools
import jax
from jax import numpy as jnp
from tallax.constants import REPLACE_VAL
from tallax.vllm.bounded_k import topp_and_sample
from tallax.tax.divide_and_filter_topk.topk import top_bounded_k


@functools.partial(
  jax.jit,
  static_argnames=(
    "max_k",
    "num_bins",
    "bins_topm_schedule",
    "debug",
    "stable",
  ),
)
def bounded_topk_topp_and_sample(
  logits,
  rng_key,
  top_k,
  top_p,
  temperature,
  max_k: int,
  num_bins: int | None = None,
  stable: bool = True,
  bins_topm_schedule: int | None = None,
  debug: bool = False,
):
  """Combined top-k, top-p filtering, and sampling for vLLM inference.

  Uses the bounded_k path: divide-and-filter top-k -> bitonic sort -> top-p -> sample.

  Args:
    rng_key: RNG key for sampling.
    logits: Input logits of shape [batch_size, vocab_size].
    tpu_sampling_metadata: Metadata containing top_k, top_p, and temperature.
    max_k: Maximum k value for top-k computation.
    num_bins: Optional number of bins for divide-and-filter algorithm.
    bins_topm_schedule: Optional custom schedule for binned top-m computation.

  Returns:
    Sampled token indices.
  """
  topk_logits, topk_idxs = top_bounded_k(
    logits,
    k=top_k,
    replace_val=REPLACE_VAL,
    max_k=max_k,
    num_bins=num_bins,
    bins_topm_schedule=bins_topm_schedule,
    guarantee_convergence=True,
    stable=stable,
  )
  outs = topp_and_sample(
    topk_logits=topk_logits,
    topk_idx=topk_idxs,
    rng_key=rng_key,
    top_p=top_p,
    temperature=temperature,
    debug=debug,
  )
  if not debug:
    return outs
  sampled, debug_vals = outs
  # Rebuild the unreduced shape arrays
  debug_vals["topk_logits_unsorted"] = jax.vmap(
    lambda ind, updates: jnp.full_like(
      updates, REPLACE_VAL, shape=logits.shape[1:]
    )
    .at[ind]
    .set(updates)
  )(debug_vals["topk_idxs"], debug_vals["topk_logits"]).astype(logits.dtype)
  debug_vals["topk_topp_unnorm_probs_i32_unsorted"] = jax.vmap(
    lambda ind, updates: jnp.zeros_like(
      updates,
      shape=logits.shape[1:],
    )
    .at[ind]
    .set(updates)
  )(
    debug_vals["topk_idxs"].sort(1),
    debug_vals["topk_topp_unnorm_probs_i32_topk_filtered_unsorted"],
  )
  del debug_vals["topk_idxs"]
  del debug_vals["topk_logits"]
  del debug_vals["topk_topp_unnorm_probs_i32_topk_filtered_unsorted"]
  return sampled, debug_vals
