# Comparison: Dr. Top-K vs Divide-and-Filter Top-K

## Executive Summary

Both **Dr. Top-K** (DrTopKSC) and **Divide-and-Filter Top-K** (tallax) implement delegate-centric/partition-based approaches to efficiently compute top-k elements, but target different hardware platforms and use different optimization strategies:

- **Dr. Top-K**: GPU-optimized (CUDA), focuses on radix/bucket select with delegate sampling
- **Divide-and-Filter**: TPU-optimized (JAX/Pallas), focuses on binned partitioning with probabilistic early stopping

Both methods achieve **substantial performance improvements** (10-99% workload reduction) over naive sorting approaches by dividing the input and filtering out partitions that cannot contribute to the final top-k.

---

## Algorithm Overview

### Dr. Top-K (Delegate-Centric on GPUs)

**Paper**: [Dr. Top-k: Delegate-Centric Top-k on GPUs](https://sc21.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap131.html) (SC'21)

**Core Concept**: Divide input into subranges, select "delegates" (representatives) from each subrange, and use these delegates to filter out subranges that cannot contribute to top-k.

**Key Components**:
1. **Subrange Division**: Input of size N divided into N/2^α subranges of size 2^α
2. **Delegate Selection**:
   - **Maximum Delegate**: Selects max value from each subrange
   - **β-Delegate**: Selects top-β values from each subrange (default β=2)
3. **Filtering**: Perform top-k on delegates to identify which subranges contain potential top-k elements
4. **Concatenation**: Merge selected subranges
5. **Final Top-K**: Apply radix select, bucket select, or bitonic sort on filtered data

**Implementations**:
- Radix Select (distribution-based, digit-by-digit refinement)
- Bucket Select (range-based bucketing)
- Bitonic Sort (comparison-based sorting network)

### Divide-and-Filter Top-K (Tallax on TPUs)

**Implementation**: [tallax divide_and_filter_topk](https://github.com/user/tallax)

**Core Concept**: Partition input into bins, incrementally compute top-m per bin until convergence criterion is met, then extract global top-k from minimal superset.

**Key Components**:
1. **Binned Partitioning**: Input divided into num_bins partitions (default 128-1024 bins)
2. **Incremental Bins-Top-M**:
   - Compute bins-top-m across all bins using "sinking sort"
   - Incrementally increase m following a schedule (e.g., m=1,2,4,8...)
3. **Convergence Check**:
   - After computing bins-top-(m+1), check if bins-top-m contains entire top-k
   - Uses threshold test: if ⌈k/m⌉-th largest m-th value across bins provides a lower bound
   - If count of values ≥ threshold in bins-top-m is ≥ k, then converged
4. **Bin Packing Optimization**: For rare non-convergence cases, pack most active bins
5. **Final Top-K**: Apply bitonic top-k on minimal superset (bins-top-m or packed bins)

---

## Technical Comparison

| **Aspect** | **Dr. Top-K (GPU/CUDA)** | **Divide-and-Filter (TPU/JAX)** |
|------------|-------------------------|--------------------------------|
| **Target Hardware** | NVIDIA GPUs (CUDA) | Google TPUs (Pallas/JAX) |
| **Implementation Language** | CUDA C++ | Python/JAX with Pallas kernels |
| **Partitioning Strategy** | Subranges with tunable α parameter | Fixed bins (128-1024) based on vocab size |
| **Delegate/Representative** | Max delegate + β-delegate (top-β per subrange) | Bins-top-m (incremental schedule) |
| **Workload Reduction** | Up to 99%+ reported | 96-99.9999% probability of early convergence |
| **Early Stopping** | Not explicitly mentioned | Probabilistic convergence checks at each m |
| **Optimization Techniques** | Shuffle optimization, digit skipping, multi-GPU | Sinking sort, bin packing, VMEM/SMEM management |
| **Memory Access Pattern** | Optimized for GPU global memory coalescing | Optimized for TPU tile-aligned access (128 lanes) |
| **Scheduling** | Static α, β parameters (tunable) | Dynamic bins_topm_schedule (auto-computed or custom) |
| **Supported Base Algorithms** | Radix Select, Bucket Select, Bitonic Sort | Bitonic Sort for final top-k |
| **Convergence Theory** | Empirical α tuning (α ≈ 0.5*(log₂N - log₂k + 3)) | Mathematical "Balls into Bins" probability theory |
| **Multi-Device Support** | Multi-GPU via MPI | Sharded computation with custom partitioning |
| **Handling Ties** | Implementation-dependent | Non-stable by default (stable option available) |
| **NaN Handling** | Not specified | Replaces NaNs with dtype minimum |

---

## Algorithmic Differences

### 1. Partitioning Philosophy

**Dr. Top-K**:
- Uses **subrange size** as the key parameter (controlled by α)
- Smaller α → larger subranges → fewer subranges → less filtering benefit
- Larger α → smaller subranges → more subranges → better filtering but higher overhead
- Formula: SubrangeSize = 2^α, where α ≈ 0.5*(log₂N - log₂k + 3)

**Divide-and-Filter**:
- Uses **number of bins** as the key parameter (typically 128-1024)
- Bins chosen to align with TPU hardware (multiples of 128 lanes)
- Adaptive bin count based on vocabulary size heuristic
- Focuses on maximizing parallel hardware utilization

### 2. Filtering Mechanism

**Dr. Top-K**:
```
1. Sample max (and top-β) from each subrange
2. Find top-k among delegates → identifies ~k/β promising subranges
3. Concatenate entire contents of promising subranges
4. Run radix/bucket/bitonic select on concatenated data
```

**Divide-and-Filter**:
```
1. Compute bins-top-m for all bins (m starts small, increases)
2. Check if max(m-th values) provides sufficient lower bound
3. If converged: top-k only the m*num_bins values
4. If not converged: increase m and repeat (or pack active bins)
5. Run bitonic top-k on minimal superset
```

### 3. Workload Reduction Strategy

**Dr. Top-K**:
- **Spatial reduction**: Identifies subranges likely to contain top-k
- **One-shot filtering**: Single delegate selection step, then concatenate
- **Fixed overhead**: Always processes β*NSubranges delegates

**Divide-and-Filter**:
- **Incremental refinement**: Gradually increases m until convergence
- **Probabilistic early exit**: Most cases converge at small m (e.g., m=4 for k=64)
- **Adaptive overhead**: Overhead grows with convergence difficulty

### 4. Convergence Guarantee

**Dr. Top-K**:
- β-delegate ensures coverage: with β≥1, selected subranges guaranteed to contain top-k
- No explicit convergence check; relies on delegate representativeness
- β=2 or β=3 typical for balancing accuracy and overhead

**Divide-and-Filter**:
- **Convergence check**: Explicitly verifies if bins-top-m contains top-k
- **Probabilistic guarantee**: Mathematical bounds on convergence probability
  - For k=128, num_bins=256:
    - Bins-top-4: >95% convergence
    - Bins-top-8: >99.9999% convergence
- **Worst-case handling**: Bin packing optimization for rare non-convergence

---

## Performance Characteristics

### Dr. Top-K (GPU)

**Strengths**:
- Excellent for large N, moderate k (e.g., N=2^29, k=2^20)
- Radix select performs well on uniform distributions
- Low constant overhead from delegate sampling
- Multi-GPU scaling demonstrated on Summit supercomputer

**Weaknesses**:
- Radix select vulnerable to skewed distributions
- Requires manual α, β tuning for optimal performance
- No built-in early stopping mechanism
- Less effective when k is very large relative to N

**Reported Performance**:
- Speedups over state-of-art: 1.2-3.5× depending on k
- Workload reduction: up to 99%+ for small k

### Divide-and-Filter (TPU)

**Strengths**:
- **Dramatic speedups for LLM inference**: 15-75× over vLLM for logit sampling
- Adaptive schedule with automatic tuning
- Probabilistic early stopping provides consistent performance
- Excellent for moderate k with large vocabularies (e.g., k=64, vocab=262K)
- Constant-time mode for very small k (e.g., k≤8)

**Weaknesses**:
- Incremental refinement adds overhead for worst-case distributions
- Requires TPU hardware (CPU fallback is slow)
- Non-stable sorting by default (ties handled differently than jax.lax.top_k)

**Reported Performance**:
- Small batch (16): 15× average speedup, 10× worst-case
- Large batch (128): 75× average speedup, 45× worst-case
- Top-5 speculative decoding: 15× over XLA

---

## Use Case Recommendations

### Choose Dr. Top-K (GPU) When:
- ✅ Working with **NVIDIA GPUs** (CUDA-based pipeline)
- ✅ Large datasets (N > 2^25) with moderate k
- ✅ Uniform or normal distributions (radix select excels)
- ✅ Multi-GPU scaling required
- ✅ Need control over base algorithm (radix/bucket/bitonic)
- ✅ Working in C++/CUDA environment

### Choose Divide-and-Filter (TPU) When:
- ✅ Working with **Google TPUs** (JAX/Pallas pipeline)
- ✅ LLM inference and logit sampling workloads
- ✅ Moderate k with very large vocabularies (e.g., k=64, vocab=256K)
- ✅ Want automatic schedule tuning and early stopping
- ✅ Batched inference with consistent k across batch
- ✅ Working in Python/JAX environment
- ✅ Need sharded computation support

---

## Algorithmic Innovations

### Dr. Top-K Innovations:
1. **β-Delegate Concept**: Selecting multiple representatives per subrange improves accuracy
2. **Shuffle Optimization**: Rearranging data for better GPU memory access patterns
3. **Digit Skipping**: In radix select, skip leading digits when range is known
4. **Unified Framework**: Same delegate concept works for radix, bucket, and bitonic

### Divide-and-Filter Innovations:
1. **Probabilistic Convergence Theory**: Mathematical framework for early stopping
2. **Sinking Sort**: Efficient incremental top-m update mechanism
3. **Bin Packing Optimization**: Handles rare non-convergence by packing active bins
4. **Custom Partitioning**: JAX custom_partitioning for distributed top-k
5. **Auto-Schedule Computation**: Automatically computes optimal bins_topm_schedule based on convergence probability

---

## Convergence Theory Comparison

### Dr. Top-K:
- **Empirical α Formula**: α ≈ 0.5*(log₂N - log₂k + 3)
- Derived from experimental results across various N and k
- Ensures NSubranges ≥ k for effective filtering
- No explicit probabilistic analysis

### Divide-and-Filter:
- **Balls into Bins Problem**: Classic combinatorics problem
- Probability that bins-top-m contains top-k computed via convolution of polynomials
- **Block-level convergence**: Uses `(CDF)^block_size` for practical batch-level guarantees
- **Automated threshold selection**: Computes m values for target yields (e.g., 0.66, 0.95, 0.9999)
- See `convergence_theory.py` for mathematical implementation

**Mathematical Insight**:
- With random distribution, most top-k elements spread across bins
- For k=128, num_bins=256: expect ~0.5 elements per bin in top-k
- Bins-top-4 likely captures all (>95%), bins-top-8 almost certain (>99.9999%)
- Early convergence is the **common case**, making incremental approach efficient

---

## Code Structure Comparison

### Dr. Top-K (CUDA):
```
DrTopKSC/
├── baseline/                  # Pure radix select
├── baseline+filter/           # + Delegate filtering
├── baseline+filter+beta/      # + β-delegate
├── baseline+filter+beta+shuffle/  # + All optimizations (final)
├── bitonic/                   # Bitonic sort variant
├── bucket_select/             # Bucket select variant
└── MultiGPU/                  # Multi-GPU implementation
```

**Key Files**:
- `main.cu`: Driver code, parameter setup
- `radixselect.cuh`: Core radix select + delegate sampling
- `run.bash`: Benchmark scripts

### Divide-and-Filter (JAX/Pallas):
```
tallax/tax/divide_and_filter_topk/
├── topk.py                    # Main implementation
├── convergence_theory.py      # Probabilistic convergence calculations
└── __init__.py
```

**Key Functions**:
- `binned_topk()`: Sinking sort for bins-top-m
- `dynamic_topk_refs()`: Pallas kernel with convergence checks
- `_merge_unconverged_bins_topk()`: Bin packing optimization
- `top_bounded_k()`: High-level interface with custom partitioning

---

## Theoretical Complexity

### Dr. Top-K (Radix Select variant):

**Without Delegates**:
- Radix Select: O(N * D) where D = number of digit passes

**With Delegates**:
- Delegate sampling: O(N)
- Top-k on delegates: O(NSubranges * β * log k)
- Concatenation: O(k/β * SubrangeSize) ≈ O(k * 2^α)
- Final radix select: O(k * 2^α * D')

**Workload Reduction**: From N to ≈ k * 2^α (often 100× or more reduction)

### Divide-and-Filter:

**Per Iteration (m)**:
- Bins-top-m: O(N) with sinking sort (m passes over vocabulary)
- Convergence check: O(num_bins)

**Expected Case** (converges at m*):
- Total: O(m* * N)
- Final bitonic top-k: O(m* * num_bins * log k)
- For k=128, m*≈4: processes ~4 * N

**Worst Case** (no convergence):
- Total: O(k * N) + bin packing overhead
- Still bounded by k rather than N

**Workload Reduction**: From N to m* * num_bins ≈ 4 * 512 = 2048 for k=64 (typical case)

---

## Hardware Utilization

### Dr. Top-K (GPU):
- **Memory coalescing**: Shuffle optimization ensures coalesced global memory access
- **Shared memory**: Uses shared memory for subrange processing
- **Warp-level primitives**: Leverages warp shuffle and warp-level reductions
- **Occupancy**: Tuned block/grid sizes for high SM occupancy
- **Multi-GPU**: MPI-based distribution for very large N

### Divide-and-Filter (TPU):
- **VMEM vs SMEM**: Strategic use of vector memory (VMEM) and scalar memory (SMEM)
- **Lane alignment**: Partitions aligned to 128 lanes for optimal vector ops
- **Tile batching**: Processes multiple tokens in tiles for efficiency
- **Block-level buffering**: Accumulates bins-top-m before final top-k for better hardware utilization
- **Custom partitioning**: Sharded top-k with all-gather for distributed inference

---

## Practical Considerations

### Dr. Top-K:

**Pros**:
- ✅ Open-source CUDA implementation readily available
- ✅ Well-tested on real hardware (Titan Xp, V100, Summit)
- ✅ Clear parameter tuning guidelines (α, β)
- ✅ Multiple algorithm variants (radix/bucket/bitonic)
- ✅ Published paper with reproducible results

**Cons**:
- ❌ CUDA-only, no CPU fallback
- ❌ Manual parameter tuning required
- ❌ Limited documentation
- ❌ No Python bindings

### Divide-and-Filter:

**Pros**:
- ✅ Python/JAX interface (easy integration)
- ✅ Automatic schedule computation
- ✅ Built-in convergence theory and probabilistic guarantees
- ✅ Sharding support for distributed workloads
- ✅ Part of larger tallax library with other optimized ops
- ✅ Extensive test suite

**Cons**:
- ❌ TPU-only (CPU interpret mode is very slow)
- ❌ Less transparent what's happening under the hood (Pallas abstraction)
- ❌ Newer implementation (less battle-tested)
- ❌ No published paper (yet) with detailed analysis

---

## Conclusion

Both **Dr. Top-K** and **Divide-and-Filter Top-K** represent significant advances in efficient top-k computation through delegate-centric/partition-based approaches. They share the **core insight** that dividing the input and filtering partitions yields massive performance gains.

**Key Takeaway**: The choice between them depends primarily on your **hardware platform** (GPU vs TPU) and **use case** (general top-k vs LLM inference sampling).

- **GPU users**: Dr. Top-K offers a mature, well-documented CUDA implementation
- **TPU users**: Divide-and-Filter provides state-of-art performance with automatic tuning

Both methods achieve the **same fundamental goal** through different optimization strategies tailored to their respective hardware architectures. The delegate-centric concept is hardware-agnostic, but implementation details matter significantly for achieving peak performance.

---

## References

### Dr. Top-K:
- **Paper**: Anil Gaihre et al., "Dr. Top-k: Delegate-Centric Top-k on GPUs", SC'21
  - [Conference Page](https://sc21.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap131.html)
  - [ResearchGate PDF](https://www.researchgate.net/publication/354780969_Dr_Top-k_Delegate-Centric_Top-k_on_GPUs)
  - [ACM DL](https://dl.acm.org/doi/10.1145/3458817.3476141)
- **Code**: [GitHub - Anil-Gaihre/DrTopKSC](https://github.com/Anil-Gaihre/DrTopKSC.git)

### Divide-and-Filter:
- **Code**: [tallax/tax/divide_and_filter_topk](https://github.com/user/tallax)
- **Delegate-Centric Concept**: Mentioned in README as "officially known as delegate centric top-k"
- **Related Work**:
  - [RadiK: Scalable and Optimized GPU-Parallel Radix Top-K Selection](https://arxiv.org/abs/2501.14336) (2025)
  - [Parallel Top-K Algorithms on GPU: A Comprehensive Study](https://dl.acm.org/doi/10.1145/3581784.3607062) (2023)

### Additional Resources:
- [Efficient Top-K Query Processing on Massively Parallel Hardware](https://www.doc.ic.ac.uk/~hlgr/pdfs/MassivelyParallelTopK.pdf) - Anil Shanbhag
- [Understanding Top-k Sparsification in Distributed Deep Learning](https://arxiv.org/abs/1911.08772)
- [Selection Algorithm (Wikipedia)](https://en.wikipedia.org/wiki/Selection_algorithm)

---

## Appendix: Parameter Recommendations

### Dr. Top-K (GPU):
| N (input size) | k | α (recommended) | β | Expected Performance |
|----------------|---|-----------------|---|---------------------|
| 2^20 | 2^10 | 12 | 2 | 2-3× speedup |
| 2^25 | 2^15 | 12 | 2 | 2-4× speedup |
| 2^29 | 2^20 | 13 | 2 | 1.5-3× speedup |

### Divide-and-Filter (TPU):
| vocab_size | k | num_bins | schedule | Expected Convergence |
|------------|---|----------|----------|---------------------|
| 2^13 | 64 | 128 | (4,) | 96% at m=4 |
| 2^15 | 128 | 256 | (4, 8) | 95% at m=4, 99.9% at m=8 |
| 2^18 | 64 | 512 | (4,) | 99% at m=4 |
| 262K | 64 | 512 | (4,) | 96%+ at m=4 (Gemini 3 Pro) |

Note: Divide-and-Filter auto-computes schedules when not specified.
