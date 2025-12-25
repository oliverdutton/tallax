# Bitonic Sort Line Profiling Summary

## Setup
- **Input Shape**: (16, 8192)
- **Stage Unroll**: 6
- **Mode**: interpret=False (compilation/tracing mode)
- **Platform**: CPU (profiling Python-level function calls during trace)

## Key Findings

### Top Time Consumers

#### 1. `bitonic_sort_arrays` (69.5 seconds total)
- **Line 432** (100% of time): Main sorting loop calling `_sort_arrays`
- This is where the actual bitonic sort stages are executed
- Expected bottleneck - contains all the sorting logic

#### 2. `_bitonic_sort_substage` (25.2 seconds total)
Most expensive operations within:
- **Line 205-207** (29.5%): Enumerate compare_and_swap in permutation path
- **Line 207** (28.5%): Computing `_compute_is_descending` for sublane comparisons
- **Line 233-234** (15.1% + 17.5%): Compare and swap for tile comparisons
- **Line 194** (4.3%): `jax.tree.map` for permutation application
- **Line 232** (4.0%): Computing left/right pairs via transpose_list_of_lists

#### 3. `_compute_is_descending` (11.5 seconds total)
- **Line 137** (71.5%): Creating bit indicator - **MAJOR BOTTLENECK**
  - `create_bit_indicator(stage, tile_start_offset + tile_local_offset + sort_dim_offset)`
  - Called 8,551 times
  - 964μs per call
- **Line 153** (27.8%): Optimized path creating bit indicator for local offset
  - `create_bit_indicator(stage, tile_local_offset + sort_dim_offset)`
  - Called 4,513 times
  - 711μs per call

#### 4. `compare_and_swap` (11.2 seconds total)
- **Line 76-77** (46.1%): Creating comparison masks
  - `_compare_pair(i, left, right)` for each array pair
  - Called 17,102 times across 8,551 invocations
- **Line 81** (13.8%): Computing ties `[(left == right) for ...]`
- **Line 95-102** (26.6%): Final `jax.tree.map` to swap values
- **Line 91-93** (13.1% total): Dynamic descending mask operations

### Utility Functions

#### 5. `to_compressed_transpose_format` (0.23 seconds)
- **Line 293** (54.4%): `jnp.split` to divide array
- **Line 294** (45.6%): `jnp.concatenate` and transpose

#### 6. `from_compressed_transpose_format` (0.13 seconds)
- **Line 301** (24.6%): Initial concatenate and transpose
- **Line 303** (53.3%): Split operation
- **Line 304** (22.1%): Final concatenate

#### 7. `_resplit` (6.9 milliseconds)
- **Line 119** (97.4%): List comprehension calling `_resplit_inner`
- Called 113 times

#### 8. `_compute_padded_shape` (72 microseconds)
- Minimal overhead - negligible impact

## Critical Paths for Optimization

### 1. **`create_bit_indicator` in `_compute_is_descending`**
- **Impact**: Consumes 71.5% of `_compute_is_descending` time (8.2+ seconds)
- **Frequency**: Called 8,551 times
- **Recommendation**: This is the #1 optimization target
  - Consider caching/memoization for repeated stage values
  - Pre-compute bit indicators for common stage values
  - Investigate if JAX compilation can optimize this better

### 2. **`compare_and_swap` mask creation**
- **Impact**: 46.1% of compare_and_swap time (5.1+ seconds)
- **Frequency**: Called 17,102 times (2x per substage invocation)
- **Recommendation**:
  - Batch mask computations where possible
  - Consider fusing mask creation with swap operations

### 3. **Permutation operations in `_bitonic_sort_substage`**
- **Impact**: ~4.3% of substage time (1.1 seconds)
- **Note**: Already using efficient `jax.tree.map` and `take_along_axis`
- **Recommendation**: Likely already well-optimized

## Stage Unroll Impact

With `stage_unroll=6`:
- First 6 stages (1-6) are unrolled and executed directly
- Remaining stages (7-13 for 8192 elements) use dynamic looping
- **Total substage calls**: 112 (confirmed by profiling hits)
  - Stages 1-6: 1 + 2 + 3 + 4 + 5 + 6 = 21 substages per slice
  - This matches the expected pattern for bitonic sort

## Function Call Hierarchy

```
bitonic_sort_arrays (69.5s)
├─ _sort_arrays (called internally)
│  ├─ to_compressed_transpose_format (0.23s)
│  ├─ _bitonic_sort_substage (25.2s) [called 112 times]
│  │  ├─ _resplit (6.9ms) [called 113 times]
│  │  ├─ _compute_is_descending (11.5s) [called 8,551 times]
│  │  │  └─ create_bit_indicator [8.2s - 71.5% of parent]
│  │  └─ compare_and_swap (11.2s) [called 8,551 times]
│  │     ├─ mask creation [5.1s - 46.1%]
│  │     ├─ tie computation [1.5s - 13.8%]
│  │     └─ jax.tree.map swap [3.0s - 26.6%]
│  └─ from_compressed_transpose_format (0.13s)
```

## Recommendations

1. **High Priority**: Optimize `create_bit_indicator` - this single function consumes ~12% of total execution time
2. **Medium Priority**: Review `compare_and_swap` mask creation - potential for batching
3. **Low Priority**: Format conversion functions are already efficient
4. **Consider**: Whether stage_unroll=6 is optimal, or if a different value would reduce compilation time while maintaining performance

## Notes on interpret=False

- This profiling captures the Python-level overhead during trace/compilation
- On TPU with `interpret=False`, these Python functions generate XLA/HLO operations
- The actual TPU execution time will have different bottlenecks (hardware operations)
- This profile is useful for understanding compilation overhead and Python-level algorithmic complexity
