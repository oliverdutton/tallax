# Final JIT Optimization Summary - Progressive Improvements

## Complete Timeline of Optimizations

### (256, 2048) - Large Scale Results

| # | Optimization | Total Time | Speedup | Tracing | C++ Lowering |
|---|--------------|------------|---------|---------|--------------|
| 0 | **Baseline (no JIT)** | 110.73s | 1.00x | 56.42s (50.9%) | 54.32s (49.1%) |
| 1 | +JIT (num_keys, has_unique_key) | 91.62s | 1.21x | 4.03s (4.4%) | 87.59s (95.6%) |
| 2 | +stage static | 67.98s | 1.63x | 6.54s (9.6%) | 61.45s (90.4%) |
| 3 | +is_descending conditional | **57.01s** | **1.94x** | 5.60s (9.8%) | 51.41s (90.2%) |

**Final Result: 1.94x faster than baseline (saved 53.72s)**

---

## Implementation Details

### Step 1: Basic JIT on compare_and_swap

```python
@functools.partial(jax.jit, static_argnames=('num_keys', 'has_unique_key'))
def compare_and_swap(lefts, rights, *, num_keys, is_descending, ...):
    ...
```

**Effect:**
- ✅ Tracing: 56.42s → 4.03s (14.0x faster!)
- ❌ C++ lowering: 54.32s → 87.59s (1.61x slower)
- ✅ Net: 110.73s → 91.62s (1.21x faster)

**Why it helps:** Reuses compiled compare_and_swap across 38,912 calls instead of tracing each one.

**Why C++ slower:** JIT boundaries create larger HLO (+28% size), more compilation overhead.

---

### Step 2: Add stage to static_argnames

```python
@functools.partial(jax.jit, static_argnames=(
  'substage', 'num_keys', 'batch_size', 'sort_dim_offset',
  'full_size', 'concat_threshold', 'max_reduce', 'stage'  # ← Added
))
def bitonic_sort_substage(arrs_tiles, *, substage, num_keys, ..., stage=None, ...):
    ...
```

**Effect:**
- ✅ Tracing: 4.03s → 6.54s (1.62x slower, but still 8.6x faster than baseline)
- ✅ C++ lowering: 87.59s → 61.45s (1.43x faster!)
- ✅ Net: 91.62s → 67.98s (1.35x faster)

**Why it helps:** `stage` is often constant (None, 0, 1, 2, etc.), so making it static allows more aggressive optimization.

**Why tracing slower:** More static variants to trace, but still way better than baseline.

---

### Step 3: Conditional routing for is_descending

```python
def _compare_and_swap_impl(lefts, rights, *, num_keys, is_descending, ...):
    """Core implementation."""
    # ... (same logic as before)

# Two JIT versions
_compare_and_swap_static = functools.partial(jax.jit,
    static_argnames=('num_keys', 'has_unique_key', 'is_descending')
)(_compare_and_swap_impl)

_compare_and_swap_dynamic = functools.partial(jax.jit,
    static_argnames=('num_keys', 'has_unique_key')
)(_compare_and_swap_impl)

def compare_and_swap(lefts, rights, *, num_keys, is_descending, ...):
    """Routes to static or dynamic version based on is_descending type."""
    is_scalar = (
        is_descending is None
        or isinstance(is_descending, (bool, int))
        or (hasattr(is_descending, 'ndim') and is_descending.ndim == 0)
    )

    if is_scalar:
        return _compare_and_swap_static(lefts, rights, ...)
    else:
        return _compare_and_swap_dynamic(lefts, rights, ...)
```

**Effect:**
- ✅ Tracing: 6.54s → 5.60s (1.17x faster)
- ✅ C++ lowering: 61.45s → 51.41s (1.19x faster)
- ✅ Net: 67.98s → 57.01s (1.19x faster)

**Why it helps:**
- When `is_descending` is scalar (common case), uses static version for better optimization
- When `is_descending` is array (uncommon case), falls back to dynamic to avoid errors
- Best of both worlds!

---

## Small Scale Results: (16, 2048)

| Optimization | Total Time | vs Baseline |
|--------------|------------|-------------|
| Baseline (no JIT) | 7.07s | 1.00x |
| Final (all optimizations) | 8.48s | 0.83x (slower) |

**Verdict:** JIT optimizations hurt small-scale performance due to overhead exceeding benefits.

**Recommendation:** Only enable for batch sizes ≥ 64-128.

---

## Key Insights

### 1. Tracing Time Dominates at Large Scale

Without JIT:
- Tracing: 56.42s (51%)
- C++ lowering: 54.32s (49%)

With final JIT:
- Tracing: 5.60s (10%)
- C++ lowering: 51.41s (90%)

**Insight:** JIT dramatically reduces tracing time (10x speedup) by reusing cached compilation.

### 2. Static Arguments Enable Better Optimization

Making arguments static allows JAX to:
- Create specialized versions for each value
- Eliminate runtime checks
- Enable more aggressive constant folding

But there's overhead:
- Must compile separate version for each static value combination
- Larger HLO graphs
- More compilation time

**Sweet spot:** Make arguments static when they have a small number of common values (like `num_keys=1,2`, `stage=None,0,1,2`).

### 3. Conditional Routing = Best of Both Worlds

The `is_descending` conditional wrapper:
- Uses static when possible (scalar values)
- Falls back to dynamic when necessary (array values)
- Avoids "unhashable type" errors
- Gets optimization benefits without breaking functionality

**This pattern can be applied to other mixed-type parameters!**

---

## Performance Summary

### (256, 2048) - Production Scale

```
Baseline:           110.73s ████████████████████████████████████████████
+JIT basic:          91.62s ████████████████████████████████████
+stage static:       67.98s ███████████████████████████
+is_desc cond:       57.01s ███████████████████████

Total improvement: 1.94x faster (saved 53.72s = 48.5%)
```

### Breakdown of Time Savings

| Source | Time Saved |
|--------|-----------|
| Python tracing | -50.82s (56.42s → 5.60s) |
| C++ lowering | -2.91s (54.32s → 51.41s) |
| **Total** | **-53.73s** |

**90% of savings come from reducing tracing time!**

---

## Recommendations

### For Development (CPU lowering)

1. **Use full optimizations** for:
   - Batch sizes ≥ 128
   - Full-scale testing
   - Performance profiling

2. **Remove JIT decorators** for:
   - Batch sizes ≤ 32
   - Fast iteration during debugging
   - Small-scale tests

3. **Medium batch sizes (32-128):**
   - Test both ways
   - Choose based on iteration speed needs

### For Production (TPU runtime)

**Strongly recommended:**
- Keep all optimizations (1.94x faster lowering)
- Especially valuable for AOT compilation
- 28% larger HLO is acceptable for 2x speedup

**Monitor:**
- Runtime performance (not just compile time)
- Memory usage (larger HLO graphs)
- Warmup time (first-call JIT compilation)

**Expected impact:**
- Compile time: 1.94x faster ✅
- Runtime: Likely neutral or slightly slower (larger graphs)
- Memory: +28% HLO size

---

## Future Optimization Ideas

### 1. Apply Conditional Routing to More Parameters

Other candidates:
- `has_unique_key` (already static, but could optimize further)
- `max_reduce` in bitonic_sort_substage
- Any bool/int parameter that's sometimes dynamic

### 2. Batch compare_and_swap Calls

Current: 38,912 separate calls
Potential: Vectorize into ~100-200 batched calls
Expected speedup: 1.3-1.5x additional

### 3. Replace Tuple Comprehensions with vmap

Bottleneck: `masks = tuple(_compare_pair(...) for ...)`
Fix: Use `jax.vmap` for parallelization
Expected speedup: 1.2-1.3x

### 4. Optimize tree.map Operations

Bottleneck: `jax.tree.map(lambda left, right: ...)`
Fix: Direct array operations where possible
Expected speedup: 1.1-1.2x

---

## Conclusion

**JIT optimizations with conditional routing achieved 1.94x speedup on (256, 2048):**

1. Basic JIT: 1.21x
2. +stage static: 1.63x
3. +is_descending conditional: 1.94x

**Key innovation:** Conditional routing based on argument type allows using static JIT when beneficial while maintaining compatibility with dynamic cases.

**Production recommendation:** Keep all optimizations for batch sizes ≥ 128. The 1.94x compile speedup is substantial and the overhead is minimal.
