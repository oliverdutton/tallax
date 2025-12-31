# Compilation Performance Tests - Quick Start Guide

## Overview

This repository contains analysis and test scripts to investigate and potentially reduce the slow compilation time for `top_bounded_k` operations on TPU.

## The Problem

**Observed**: (256, 2048) takes ~16x longer to compile than (16, 2048)
- (16, 2048): 16.49s total
- (256, 2048): 254.36s total

**Key Finding**: The slowdown is NOT due to grid size (number of programs), but due to larger scratch buffer sizes and array dimensions that scale compilation complexity.

## Test Scripts

### 1. Baseline Analysis

**File**: `analyze_compilation.py`

Analyzes grid configuration and buffer sizes without running any code.

```bash
python analyze_compilation.py
```

**Output**: Grid size, buffer dimensions, and memory usage for both shapes.

---

### 2. Optimization Barriers

**File**: `test_optimization_barriers.py`

Tests if `jax.lax.optimization_barrier` reduces compilation time by preventing fusion across kernel sections.

```bash
python test_optimization_barriers.py
```

**What it does**:
- Adds barriers after initialization
- Adds barriers between binned_topk and storage
- Adds barriers before convergence checking
- Adds barriers between major sections

**Expected**: 1.5-2x faster compilation (optimistic: 5-6x)

---

### 3. Named Scopes

**File**: `test_named_scope.py`

Tests if `jax.named_scope` helps organize compilation.

```bash
python test_named_scope.py
```

**What it does**:
- Wraps major sections in named scopes
- Improves HLO readability
- May help compiler cache decisions

**Expected**: Minimal compilation time change, but much better HLO dumps

---

### 4. Reduced Loop Unrolling

**File**: `test_unroll_reduction.py`

Tests different `bins_topm_unroll` values to reduce jaxpr creation time.

```bash
python test_unroll_reduction.py
```

**What it does**:
- Tests unroll values: 8, 16, 32, 64 (default)
- Measures compilation time for each
- Shows tradeoff between compile time and runtime

**Expected**: 2-3x faster jaxpr creation with unroll=16 vs 64

---

### 5. Grid Impact Analysis (CPU only)

**File**: `analyze_grid_impact.py`

Attempts to analyze jaxpr complexity on CPU.

```bash
python analyze_grid_impact.py
```

**Note**: May fail due to jitted function limitations. Informational only.

---

## Quick Test: Most Promising Approach

If you want to test the most promising optimization quickly:

```bash
python test_unroll_reduction.py
```

This tests reduced loop unrolling, which is most likely to show significant improvement.

## Expected Results Summary

| Technique | Jaxpr Time | Lowering Time | Compilation Time | Overall |
|-----------|------------|---------------|------------------|---------|
| Baseline | Baseline | Baseline | Baseline | Baseline |
| Barriers | ~Same | 1.1-1.3x faster | 1.1-1.2x faster | 1.1-1.5x faster |
| Named Scope | ~Same | ~Same | ~Same | ~Same |
| Unroll=32 | 1.5-2x faster | ~Same | ~Same | 1.2-1.5x faster |
| Unroll=16 | 2-3x faster | ~Same | ~Same | 1.5-2x faster |
| Combined | 2-3x faster | 1.2x faster | 1.1x faster | **1.5-2.5x faster** |

## Best Combined Approach

Test all three together for maximum effect:

1. Use `bins_topm_unroll=16` or `32` instead of default `64`
2. Add optimization barriers at strategic points
3. Wrap major sections in named scopes

**Expected Total Speedup**: 1.5-2.5x compilation time reduction

## Recommended Testing Order

1. **Start here**: `test_unroll_reduction.py` - Easiest to implement, likely biggest impact
2. **Then try**: `test_optimization_barriers.py` - More complex, may help
3. **Finally**: `test_named_scope.py` - Minimal time impact, but better debugging

## Understanding the Results

### If you see 2x+ speedup:
Great! The technique works. Consider using it in production.

### If you see 1.2-1.5x speedup:
Good! Worthwhile if compilation time is a bottleneck.

### If you see <1.1x speedup:
The technique doesn't help much. Try another approach or accept the compilation time.

### If runtime gets slower:
The optimization barriers prevented beneficial fusions. Balance compile time vs runtime.

## Alternative Approaches (if barriers don't help)

1. **AOT Compilation**: Pre-compile common shapes and cache
2. **Larger block_token**: Process more tokens per program
3. **Split the kernel**: Break into multiple Pallas calls
4. **Accept it**: 13-16x slowdown for 16x larger input might be fundamental

## Files Reference

### Documentation
- `COMPILATION_ANALYSIS.md` - Initial (incorrect) analysis
- `CORRECTED_ANALYSIS.md` - Corrected analysis of actual causes
- `BARRIER_TECHNIQUES.md` - Detailed guide on barrier techniques
- `RECOMMENDED_TESTS.md` - Tests to run on TPU hardware
- `README_COMPILATION_TESTS.md` - This file

### Test Scripts
- `analyze_compilation.py` - Analyze grid and buffer configurations
- `analyze_grid_impact.py` - Analyze jaxpr complexity (CPU)
- `test_optimization_barriers.py` - Test optimization barriers
- `test_named_scope.py` - Test named scopes
- `test_unroll_reduction.py` - Test reduced loop unrolling
- `debug_compile_timing.py` - Detailed timing breakdown
- `test_tpu_lowering_cpu_v2.py` - Attempt TPU lowering on CPU (fails)

### Analysis Files (from earlier work)
- `debug_compile_timing.py` - Original timing tool
- `test_tpu_lowering_on_cpu.py` - Early TPU lowering attempt

## How to Apply Optimizations to Your Code

If tests show promising results, here's how to modify your actual code:

### Apply Reduced Unroll

```python
# In your code that calls top_bounded_k:
result = top_bounded_k(
    logits,
    k,
    max_k=128,
    bins_topm_unroll=32,  # Changed from default 64
    ...
)
```

### Apply Barriers (requires modifying tallax source)

Edit `tallax/tax/divide_and_filter_topk/topk.py` and add barriers as shown in `test_optimization_barriers.py`.

### Apply Named Scopes

Edit `tallax/tax/divide_and_filter_topk/topk.py` and wrap sections in `with jax.named_scope(...)` as shown in `test_named_scope.py`.

## Questions?

See the detailed documentation files for more information:
- **What's the root cause?** → `CORRECTED_ANALYSIS.md`
- **How do barriers work?** → `BARRIER_TECHNIQUES.md`
- **What else can I try?** → `RECOMMENDED_TESTS.md`

## Summary

The 16x compilation slowdown is likely fundamental to processing 16x more data, but these techniques may reduce it to ~8-10x:

1. **Reduce loop unrolling** (most promising)
2. **Add optimization barriers** (moderate promise)
3. **Use named scopes** (minimal compile time impact, better debugging)

Test these on your TPU hardware to see actual impact!
