"""Bounded-k sampling path using sorting.

This code path first applies top-k via bitonic sort to reduce the input
to k elements, then applies top-p and sampling on the sorted subset.

Suitable when k is small (typically <= 128) since sorting cost is O(k log^2 k).
The top-k itself uses the divide-and-filter algorithm which is stable when
enabled.
"""

from tallax.vllm.bounded_k.top_p_and_sample import bounded_topk_topp_and_sample

__all__ = ["bounded_topk_topp_and_sample"]
