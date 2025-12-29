"""Tallax vLLM integration module.

Public API for vLLM-compatible sampling operations.
"""

from tallax.vllm.sampling import topk_topp_and_sample
from tallax.vllm.sampling import top_p_and_sample

__all__ = [
  "topk_topp_and_sample",
  "top_p_and_sample",
]
