"""Full-vocabulary sampling path using binary search (no sorting).

This code path keeps the full vocabulary size and uses binary searches
to find top-k and top-p thresholds without ever sorting the logits.

Suitable for large vocabularies where sorting would be prohibitively expensive.
"""

from tallax.vllm.arbitrary_k.kernel import topk_topp_mask_and_sample

__all__ = ["topk_topp_mask_and_sample"]
