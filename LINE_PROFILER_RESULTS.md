# Line Profiler Results - Bottleneck Analysis

## Summary

Used line_profiler on `.lower()` call to identify exact lines causing lowering slowness.

**Test**: (16, 2048) shape
**Total lowering time**: ~16-18s on CPU

## Top Functions by Total Time

| Function | File | Total Time | % |
|----------|------|------------|---|
| `bitonic_sort_substage` | tallax/tax/bitonic/sort.py:253 | **8.30s** | **50%** |
| `compare_and_swap` | tallax/tax/bitonic/sort.py:34 | **6.16s** | **37%** |
| JAX internals | jax/_src/traceback_util.py | 6.82s | 41% |

**Total in bitonic functions: 14.46s (~86% of lowering time)**

## Detailed Breakdown

### 1. `bitonic_sort_substage` (8.30s total)

**Hot lines:**

| Line | Code | Time | % |
|------|------|------|---|
| **314** | `compare_and_swap(lefts, rights, ...)` | **4.60s** | **55.4%** |
| **301** | `jax.tree.map(lambda tile: jnp.take_along_axis(...))` | **0.89s** | **10.8%** |
| **317** | `_compute_is_descending(...)` | **0.55s** | **6.7%** |
| 295 | `tile_local_offset = iota_tile(0) + ...` | 0.11s | 1.4% |
| 296-298 | `create_bit_indicator(...)` | 0.05s | 0.6% |
| 349 | `iota_tile(1, tile_shape) // batch_size` | 0.05s | 0.5% |

**Key finding**: The call to `compare_and_swap` at line 314 accounts for **55.4% of this function's time** (4.6s out of 8.3s).

**Code at line 314:**
```python
for arr_idx, out in enumerate(
    compare_and_swap(
        lefts,
        rights,
        is_descending=_compute_is_descending(...),
        is_right_half=is_right_half,
        num_keys=num_keys,
    )
):
```

This is called **3,520 times** (1,760 iterations * 2 outputs per call).

### 2. `compare_and_swap` (6.16s total)

**Hot lines:**

| Line | Code | Time | % |
|------|------|------|---|
| **83-86** | `masks = tuple(_compare_pair(i, left, right) for ...)` | **3.15s** | **51.2%** |
| **102** | `jax.tree.map(lambda left, right: jnp.where(mask, ...))` | **1.79s** | **29.1%** |
| **88** | `ties = [(left == right) for left, right in ...]` | **0.84s** | **13.6%** |
| 98-100 | Dynamic descending mask operations | 0.36s | 5.9% |

**Code at line 83-86** (creating comparison masks):
```python
masks = tuple(
    _compare_pair(i, left, right)
    for i, (left, right) in enumerate(zip(lefts, rights, strict=True))
)
```

Called **2,432 times** during lowering.

**Code at line 102** (applying masks with jax.tree.map):
```python
return jax.tree.map(
    lambda left, right: (
        (jnp.where(mask, left, right), jnp.where(mask, right, left))
        if is_right_half is None
        else jnp.where(mask, left, right)
    ),
    lefts,
    rights,
)
```

## Root Cause Analysis

### Why is this slow during lowering?

During `.lower()`, JAX is **tracing** the computation to build the HLO (High-Level Operations) graph. Each of these operations creates nodes in the computation graph:

1. **Tuple comprehensions** (line 83-86): Creates separate graph nodes for each comparison
2. **jax.tree.map** (lines 301, 102): Traces through the tree structure, creating nodes for each leaf
3. **List comprehensions** (line 88): Creates separate graph nodes for each equality check

The problem is **not runtime execution** - it's the **graph construction complexity** during lowering.

### Key Insight

The bitonic sort implementation uses:
- Nested loops that call `compare_and_swap` 3,520 times
- Each call creates tuple comprehensions and tree maps
- This creates a **massive computation graph** during tracing

For (256, 2048):
- 16x larger buffer
- Proportionally more calls to these functions
- ~16x longer lowering time

## Potential Optimizations

### 1. Reduce calls to `compare_and_swap`

**Current**: Called 3,520 times for (16, 2048)

**Possible**: Batch comparisons to reduce call count

**Impact**: Could reduce line 314 time (4.6s) by ~50-70%

### 2. Optimize tuple/list comprehensions

**Current**: Lines 83-86 use tuple comprehension

**Alternative**: Use `jax.vmap` or vectorized operations instead of comprehensions

**Impact**: Could reduce line 83-86 time (3.15s) by ~30-50%

### 3. Simplify `jax.tree.map` calls

**Current**: Lines 102 and 301 use `jax.tree.map` with lambda functions

**Alternative**: Direct array operations when possible

**Impact**: Could reduce lines 102 + 301 time (2.68s) by ~20-40%

### 4. Use compile flags to reduce graph complexity

Already tested: Disabling XLA passes gives 1.42x speedup

**Additional**: Could explore JAX compilation flags to reduce graph size

## Recommendations

### Immediate (for (256, 2048) test):

1. ✅ **Remove nested `pl.when`** - Already implemented (1.79x speedup)
2. ✅ **Disable XLA passes** - Already tested (1.42x speedup)

### Medium-term (requires code changes):

3. **Batch `compare_and_swap` calls** - Reduce from 3,520 to ~100-200 calls
4. **Replace comprehensions with vmap** - In compare_and_swap function
5. **Simplify tree.map operations** - Use direct array ops where possible

### Long-term (architectural):

6. **Cache lowered functions** - Pre-compile common shapes
7. **Alternative bitonic implementation** - Explore different algorithms
8. **JIT subfunctions** - Already tested, gives 1.06x speedup

## Expected Combined Impact

| Optimization | Expected Speedup | Cumulative |
|--------------|------------------|------------|
| Remove nested pl.when | 1.79x | 1.79x |
| Disable XLA passes | 1.42x | 2.54x |
| Batch compare_and_swap | 1.5x | 3.81x |
| Optimize comprehensions | 1.3x | 4.95x |

**Best case**: ~5x total speedup on lowering time

**Realistic**: ~2.5-3x speedup with immediate + medium-term optimizations

## Files Analyzed

- `/home/user/tallax/tallax/tax/bitonic/sort.py` - **Main bottleneck**
- `/home/user/tallax/tallax/tax/bitonic/topk.py` - Calls bitonic_sort functions
- `/home/user/tallax/tallax/tax/bitonic/__init__.py` - Entry point

## Line Profiler Output

Full results saved to: `/tmp/line_profiler_16_2048.txt`

Most time spent in:
1. `compare_and_swap`: tallax/tax/bitonic/sort.py:34
2. `bitonic_sort_substage`: tallax/tax/bitonic/sort.py:253

Total profiled time in bitonic module: **14.46 seconds** out of ~16-18s total lowering time.

## Next Steps

1. Wait for (256, 2048) simplified extraction test to complete
2. Verify 1.79x speedup from removing nested pl.when
3. Consider batching/vmap optimizations for compare_and_swap
4. Profile on TPU hardware to validate CPU findings
