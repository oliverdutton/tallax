"""Tallax vLLM integration module.

Public API for vLLM-compatible sampling operations.

Two code paths:
  - fullvocab: Binary-search-based top-k/top-p on the full vocabulary (no sorting).
  - reducedk: Bitonic-sort-based top-k reduction then top-p on the small sorted subset.
  - reference: Pure JAX reference implementation (no Pallas).
"""

from tallax.vllm.sampling import topk_topp_and_sample
from tallax.vllm.reducedk import top_p_and_sample
from tallax.vllm.fullvocab import topk_topp_mask_and_sample
from tallax.vllm.reference import reference_topk_topp_mask_and_sample

__all__ = [
  "topk_topp_and_sample",
  "top_p_and_sample",
  "topk_topp_mask_and_sample",
  "reference_topk_topp_mask_and_sample",
]
