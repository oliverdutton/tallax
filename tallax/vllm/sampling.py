"""
vLLM top-k top-p sampling, using two pallas functions
"""

from tallax.vllm.top_p_and_sample import top_p_and_sample
from tallax.tax.divide_and_filter_topk.topk import top_bounded_k


def topk_topp_and_sample(
  rng_key,
  logits,
  tpu_sampling_metadata,
  max_k=128,
  num_bins=256,
  bins_topm_schedule=(5, 9),
):
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
