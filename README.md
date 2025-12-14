# tallax

`tallax` provides high-performance sorting and top-k operations for JAX, optimized for TPUs using Pallas.

## Installation

The installation process for `tallax` depends on your JAX backend (CPU or TPU). For `tallax` to function correctly, you must first install the appropriate version of JAX for your hardware.

### 1. Install JAX

Follow the [official JAX installation guide](https://github.com/google/jax#installation) to install `jax` and `jaxlib` for your specific accelerator (CPU, GPU, or TPU).

For example, to install JAX for a TPU environment, you might run:
```bash
pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### 2. Install tallax

Once JAX is installed, you can install `tallax` using pip:

```bash
pip install .
```

If you installed `jax[tpu]`, the `tallax` installation will automatically use it. Otherwise, it will use the CPU version of JAX.


# Divide and Filter Top-K Algorithm

A TPU-optimized algorithm for efficiently finding top-k elements through partitioning, parallel local top-m computation, and opportunistic early stopping.

---

## Overview

The algorithm finds top-k elements by:

1. **Partitioning** the input
2. **Computing top-m** for each partition in parallel
3. **Identifying unconverged partitions** — those where values beyond their top-m could still be part of the overall top-k[^0]
4. **Running top-k** over only the top-m values and unconverged partitions contents

This divide-and-filter approach dramatically reduces the amount of elements to compute top-k on.

[^0]: The ⌈k/m⌉'th largest value across the m'th largest value in each partition is a lower bound for the top-k threshold, as in ⌈k/m⌉ bins there are at least m values larger or equal to it (⌈k/m⌉ is the ceiling division of k by m). All partitions where the m'th largest value is less than the threshold will not contribute any further values to top-k so only ⌈k/m⌉-1 partitions could possibly contribute to top-k beyond their top-m.

---

## Early Stopping

The algorithm exploits probabilistic convergence for significant speedups. For randomly partitioned inputs with 256 bins, collectively bins-top-4 has a >95% probability of containing the entire top-128, rising to >99.9999% by bins-top-8.[^1] [^2] Checking for convergence and top-k'ing the minimal number of elements significantly improves average runtimes.

[^1]: The convergence theory is a classic "Balls into Bins" combinatorics problem — probability calculation code is included in tallax.

[^2]: TPU hardware utilization is often optimal with batch size of 8 as a minimal unit, so `probability^batch_dim` is the more practical value, reducing probabilities to 70% and 99.9995% respectively

### Convergence Check

Rather than running larger m values unconditionally, the algorithm checks for convergence at each step:

1. Compute **bins-top-(m+1)** instead of just bins-top-m
2. Take the **maximum (m+1)th value** across all bins — this is the largest possible value *not* in bins-top-m
3. Count how many values in bins-top-m are **≥ this threshold**
4. If **count ≥ k**, then bins-top-m contains the entire top-k

This check adds minimal overhead: just a single max and a single sum across bins — no top-k operation on the bins-top-m required.

---

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

For **k ≤ 128**, top-k can be computed in this format without the compression adding lane permute operations.

> **Result:** This Pallas implementation is significantly faster than XLA's top-k and the naive transpose implementation which uses excessive padding for small batch sizes.

### Supported Configurations

Currently implemented: **k ≤ 128** (covers most typical LLM top-k usage)

Theoretically extensible to larger k with larger sort dimensions with low batch size while still avoiding excessive lane permutations/transposes with compressed transpose format:
- k = 256 with `(8, 4096)` or `(16, 2048)`

---

## Why Use Bitonic Top-k in the Algorithm

For computing top-128 of a typical LLM vocabulary size (100–200k tokens) using the divide and filter top-k tactic with 256 bins, we expect a 99.9999% chance of convergence by bins-top-8. This produces a filtered subset size of only 256 × 8 = 2,048 elements.

At this reduced size, the choice of final top-k algorithm (Bitonic vs. RadixSelect, etc.) contributes negligibly to overall runtime, so alternatives to bitonic top-k were not explored.​​​​​​​​​​​​​​​​