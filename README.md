# tallax
`tallax` provides high-performance top-k, approx top-k, sort and gather operations for JAX, optimized for TPUs using Pallas.

Built on the lightning fast top-k a highly optimized vLLM top-k top-p logit sampler is provided.

## 🔥 Performance Wins

### 🎯 Scenario 1: Logit sampling

```
📊 Setup: Gemini 3 Pro decoding
  Top-k=64 | Top-p=0.95 | Vocab=262K | bfloat16*
```

#### 📦 Small Batch (16)

```
vLLM    ████████████████████████ 390μs
tallax  ██ 35μs
         
     🔥 10× AVERAGE SPEEDUP
     ⚡ 6× WORST-CASE SPEEDUP (70μs)
```

#### 📦📦📦 Large Batch (128)

```
vLLM    ████████████████████████████████ 11,800μs
tallax  █ 250μs

     🔥 45× AVERAGE SPEEDUP
     ⚡ 23× WORST-CASE SPEEDUP (500μs)
```

### 🎯 Scenario 2: Speculative Decoding Top-k
```
📊 Setup: Top-5 | Batch=16 | Draft-Vocab=32K | bfloat16

XLA     ████████████████████ 85μs
tallax  █ 5.5μs

     🔥 15× FASTER
```

**Gemini 3 Pro uses [fixed top-k=64 and default top-p=0.95](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/3-pro). Vocab size is not specified, so we use the Gemma 3 vocab size of 262K, logits dtype is not specified but bfloat16 is most likely.*

----


## Installation

You can install `tallax` using pip:

```bash
pip install .[tpu]
```

----

# Divide and Filter Top-k Algorithm

Tallax provides a TPU-optimized algorithm for efficiently finding top-k elements through partitioning, parallel local top-m computation, and opportunistic early stopping.



## Overview

The algorithm finds top-k elements by:

1. **Partitioning** the input
2. **Computing top-m** for each partition in parallel
3. **Identifying unconverged partitions** — those where values beyond their top-m could still be part of the overall top-k[^0]
4. **Running top-k** over only the top-m values and unconverged partitions contents

This divide-and-filter approach dramatically reduces the amount of elements to compute top-k on.

[^0]: The ⌈k/m⌉'th largest value across the m'th largest value in each partition is a lower bound for the top-k threshold, as in ⌈k/m⌉ bins there are at least m values larger or equal to it (⌈k/m⌉ is the ceiling division of k by m). All partitions where the m'th largest value is less than the threshold will not contribute any further values to top-k so only ⌈k/m⌉-1 partitions could possibly contribute to top-k beyond their top-m.

## Early Stopping

The algorithm exploits probabilistic convergence for significant speedups. For randomly partitioned inputs with 256 bins, collectively bins-top-4 has a >95% probability of containing the entire top-128, rising to >99.9999% by bins-top-8.[^1] [^2] Checking for convergence and top-k'ing the minimal number of elements significantly improves average runtimes.

[^1]: The convergence theory is a classic "Balls into Bins" combinatorics problem — probability calculation code is included in tallax. For top-k'ing LLM logits, as tokenizers often have the first indices as the most likely token then decreasing by construction you would expect even faster convergence than the random distribution assumption used here.

[^2]: TPU hardware utilization is often optimal with batch size of 8 as a minimal unit, so `probability^batch_dim` is the more practical value, reducing probabilities to 70% and 99.9995% respectively

### Convergence Check

We run a convergence check to see if bins-top-m covers top-k using bounds from bins-top-(m+1):

1. Compute **bins-top-(m+1)** instead of just bins-top-m
2. Take the **maximum (m+1)th value** across all bins — this is the largest possible value *not* in bins-top-m
3. Count how many values in bins-top-m are **≥ this threshold**
4. If **count ≥ k**, then bins-top-m contains the entire top-k

This check adds minimal overhead: just a single max and a single sum across bins — no top-k operation on the bins-top-m required.

## TPU Optimization

### Tile-Aligned Partitioning

When the number of partitions is a **multiple of 128**, the algorithm becomes highly efficient:

- Finding each partition's top-m involves only **full-tile comparisons**
- Unconverged partitions can be gathered via a **single lane permute per tile**

### Bitonic Top-K Implementation

Bitonic sorting is well-suited for highly parallel hardware like TPUs, but naive implementations suffer from excessive lane permutations (very slow on TPU).

#### The Transpose Optimization

Instead of sorting along the lane axis:

1. **Transpose** from `(batch_dim, sort_dim)` to `(sort_dim, batch_dim)`
2. **Sort along sublane axis** — sublane permutations are faster and fewer permutations overall are required (as the tile sublane size is 8 instead of 128)

**Problem:** In transposed format, `batch_dim` is padded to 128. For `batch_dim = 8`, hardware utilization drops to 1/16th.

#### Compressed Transpose Format

To recover efficiency:

1. **Distribute** `sort_dim` across both dimensions
2. **Example:** `(8, 2048)` → split into 16 tiles of `(8, 128)` → concatenate to `(128, 128)` → transpose

For **k ≤ 128**, top-k can be computed in this format with at most 4 sequential lane permute operations.

**Result:** This Pallas implementation is significantly faster than XLA's top-k.

## Why Use Bitonic Top-k in the Algorithm and not alternatives?

For computing top-128 of a typical LLM vocabulary size (100–200k tokens) using the divide and filter top-k tactic with 256 bins, we expect a 99.9999% chance of convergence by bins-top-8. This produces a filtered subset size of only 256 × 8 = 2,048 elements.

At this reduced size, the choice of final top-k algorithm (Bitonic vs RadixSelect vs ...) contributes negligibly to overall runtime , so alternatives to bitonic top-k were not explored.​​​​​​​​​​​​​​​​

## How does this compare to jax.lax.approx_max_k[^8]?
- As the name says, approx_max_k is just an approximation with weak guarantees, it can miss the 2nd largest element. While tallax top-k guarantees exactness.
- **tallax top-k is a generalization** of the approx_max_k algorithm. In approx_max_k the input is split into bins which are top-1'd, before a top-k on the aggregate. Tallax generalizes this to top-m instead of top-1, adding early stopping with convergence checks and an efficient convergence bounds based method for reducing worst case runtime.
- As tallax is a generalization, we provide **tallax.tax.approx_max_k**. Our implementation can be **up to 5x faster** than jax.lax.approx_max_k due to a more efficient bitonic top-k on the aggregated top-1's. See table below:


| Shape | k=128 (b=16) | k=64 (b=16) | k=16 (b=16) | k=128 (b=128) | k=64 (b=128) | k=16 (b=128) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| (b, 1024) | **2.9x** | **3.3x** | **2.8x** | 1.2x | 1.3x | 1.0x |
| (b, 32768) | **5.2x** | **3.2x** | **2.1x** | **2.8x** | **2.2x** | 1.2x |
| (b, 262144) | **2.1x** | **1.5x** | 1.2x | 1.5x | 1.2x | 1.0x |

**Note: Speedups >1.5x are in bold. Recall target kept at it's default value of 0.95. All inputs are bfloat16 and runtimes on v5e.*
- Our implementation is open source so external users can inspect and debug the code. [The Anthropic team found a significant bug in approx_max_k](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues) where for k=256, N=12000 with all zeros except a single 1 that if the 1 was beyond index 10240 it would be missed. The XLA-TPU code is closed source so could not be investigated. The bug is caused by the algorithms calculation of number of bins to top-k, (k-1)/(1-recall_target=0.95)=5100, this gets rounded to hardware aligned multiple of 128 to 5120. Splitting the input to three parts: 0:5120, 5120:10240, 10240:12000, of which that sublength remainder of 1760 elements does not appear to be getting compared. Having access to the source code makes this bug far more easier to spot. tallax binned topk you can see [here](https://github.com/oliverdutton/tallax/blob/f6805bfccd23613129a864381fda3feaa6f05230/tallax/_src/divide_and_filter_topk.py#L96-L104) does handle this remainder portion correctly, padding the final remainder to the number of bins and comparing.

[^8]: [TPU-KNN
K Nearest Neighbor Search at Peak FLOP/s](https://arxiv.org/pdf/2206.14286)

----
