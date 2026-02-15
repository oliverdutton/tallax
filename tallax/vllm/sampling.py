"""vLLM top-k top-p sampling, using two stages.

Stage 1 (reducedk path): Divide-and-filter top-k to reduce vocab to k elements.
Stage 2 (reducedk path): Top-p + sample on the reduced sorted subset.

For direct full-vocab binary-search path, use fullvocab.topk_topp_mask_and_sample.
"""

import functools
import jax
from tallax.vllm.reducedk import top_p_and_sample
from tallax.tax.divide_and_filter_topk.topk import top_bounded_k


@functools.partial(
  jax.jit, static_argnames=("max_k", "num_bins", "bins_topm_schedule")
)
def topk_topp_and_sample(
  rng_key,
  logits,
  tpu_sampling_metadata,
  max_k: int,
  num_bins: int | None = None,
  bins_topm_schedule: int | None = None,
):
  """Combined top-k, top-p filtering, and sampling for vLLM inference.

  Uses the reducedk path: divide-and-filter top-k -> bitonic sort -> top-p -> sample.

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
  vocab_size = logits.shape[1]
  topk_logits, topk_idxs = top_bounded_k(
    logits,
    k=tpu_sampling_metadata.top_k,
    replace_val=-1e12,
    max_k=max_k,
    num_bins=num_bins,
    bins_topm_schedule=bins_topm_schedule,
    guarantee_convergence=True,
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
