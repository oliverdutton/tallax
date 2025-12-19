# tallax
`tallax` provides high-performance top-k, sort and gather operations for JAX, optimized for TPUs using Pallas.

Built on the lightning fast top-k a highly optimized vLLM top-k top-p logit sampler is provided.

## 🔥 Performance Wins

### 🎯 Scenario 1: Logit sampling

```
📊 Setup: Gemini 3 Pro decoding
  Top-k=64 | Top-p=0.95 | Vocab=262K | bf16*
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
📊 Setup: Top-5 | Batch=16 | Vocab=32K | bf16

XLA     ████████████████████ 85μs
tallax  █ 5.5μs

     🔥 15× FASTER
```

**Gemini 3 Pro uses fixed top-k=64 and default top-p=0.95. Vocab size is not specified, so we use the Gemma 3 vocab size of 262K, logits dtype is not specified but bfloat16 is most likely.*

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

### Supported Configurations

Currently implemented: **k ≤ 128** (covers most typical LLM top-k usage)

Theoretically extensible to larger k with larger sort dimensions with low batch size while still avoiding excessive lane permutations/transposes with compressed transpose format:
- k = 256 with `(8, 4096)` or `(16, 2048)`

## Why Use Bitonic Top-k in the Algorithm and not alternatives?

For computing top-128 of a typical LLM vocabulary size (100–200k tokens) using the divide and filter top-k tactic with 256 bins, we expect a 99.9999% chance of convergence by bins-top-8. This produces a filtered subset size of only 256 × 8 = 2,048 elements.

At this reduced size, the choice of final top-k algorithm (Bitonic vs RadixSelect vs ...) contributes negligibly to overall runtime , so alternatives to bitonic top-k were not explored.​​​​​​​​​​​​​​​​

----
