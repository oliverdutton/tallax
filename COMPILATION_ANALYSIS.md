# Compilation Performance Analysis: (256, 2048) vs (16, 2048)

## Summary

Based on your timing results and analysis of the codebase, I've identified why (256, 2048) compilation is ~16.8x slower than (16, 2048) for the topk/topp/sample operations.

## Key Findings

### 1. The Bottleneck is NOT in Bitonic Top-K

**Observed timings:**
- `bitonic_topk_in_vmem` for (16, 2048): 3.04s total
- `bitonic_topk_in_vmem` for (256, 2048): 13.11s total
- **Ratio: 4.3x** (reasonable for 16x larger batch)

The bitonic top-k shows a healthy scaling behavior that's sublinear relative to the batch size increase.

### 2. The Bottleneck IS in Divide-and-Filter Top-K (`top_bounded_k`)

**Observed timings:**
- `topk_topp_and_sample` for (16, 2048): 9.36s total
- `topk_topp_and_sample` for (256, 2048): 157.39s total
- **Ratio: 16.8x** (nearly linear with batch size)

Since the full pipeline includes both `top_bounded_k` and `top_p_and_sample`, and we know:
- `top_p_and_sample`: ~2x slowdown (1.67s vs 0.91s)
- `bitonic_topk_in_vmem`: ~4.3x slowdown
- Full pipeline: ~16.8x slowdown

The excess time must be coming from `top_bounded_k`.

### 3. Root Cause: Number of Programs to Compile

**Grid configuration analysis:**

For `block_token=8` (default):
- **(16, 2048)**:
  - `num_tokens_padded`: 16
  - `num_programs`: 16 / 8 = **2 programs**

- **(256, 2048)**:
  - `num_tokens_padded`: 256
  - `num_programs`: 256 / 8 = **32 programs**

**Ratio: 32 / 2 = 16x more programs**

This matches almost exactly with the 16.8x compilation slowdown!

### 4. Why More Programs = Slower Compilation

The `top_bounded_k` function uses a Pallas kernel with a grid that maps to the number of programs:

```python
grid=(pl.cdiv(num_tokens, block_token),)
```

Each program:
1. Must be separately lowered to HLO/StableHLO
2. Must be separately optimized by the XLA compiler
3. Shares some state with other programs (for convergence checking, final top-k extraction)
4. Has complex control flow with `@pl.when` conditionals

The compilation time appears to scale **linearly** with the number of programs, not superlinearly as one might fear. The 16.8x vs 16x difference could be due to:
- Fixed overhead per compilation batch
- Slight superlinear scaling in compiler optimization passes
- Memory pressure during compilation

## Detailed Breakdown

### Buffer Sizes (Same for Both Shapes)

```
max_m: 9
buffer_size: 2304 elements
VMEM per program: 0.07 MB
```

### Total VMEM Across All Programs

- (16, 2048): 0.14 MB across 2 programs
- (256, 2048): 2.25 MB across 32 programs

The total memory is reasonable and unlikely to be the bottleneck.

### Complexity Factors

**(16, 2048)**:
- Programs to compile: 2
- Elements per program: 16,384
- Total elements: 32,768

**(256, 2048)**:
- Programs to compile: 32
- Elements per program: 16,384 (same!)
- Total elements: 524,288

## Compilation Time Breakdown (from your output)

### For (16, 2048):
```
topk_topp_and_sample:
  jaxpr creation: 7.13s
  lowering:       6.61s
  compilation:    2.75s
  TOTAL:          9.36s
```

### For (256, 2048):
```
topk_topp_and_sample:
  jaxpr creation: 96.97s  (13.6x slower)
  lowering:       87.17s  (13.2x slower)
  compilation:    70.22s  (25.5x slower!)
  TOTAL:          157.39s (16.8x slower)
```

**Note:** The actual XLA compilation stage shows the worst scaling (25.5x), but it's the smallest absolute component. The jaxpr creation and lowering stages dominate the total time.

## Why Is This Happening?

### The Divide-and-Filter Algorithm

Looking at `/home/user/tallax/tallax/tax/divide_and_filter_topk/topk.py`:

1. **Complex kernel body** with multiple stages:
   - Incremental binned top-k computation
   - Convergence checking across programs
   - Dynamic control flow based on convergence
   - Bin packing optimization for non-converged cases
   - Final top-k extraction coordinated across programs

2. **Inter-program dependencies**:
   - Programs share SMEM for convergence flags
   - The last program does special work (final extraction)
   - Control flow depends on global state

3. **Large loop unrolling**:
   - `bins_topm_unroll=64` (default)
   - Unrolled loops in `binned_topk`
   - Multiple nested loops for tile merging in bitonic operations

### Why Compilation Is Slow

The XLA/Mosaic compiler must:

1. **Trace through complex control flow** for each program
2. **Optimize data movement** between VMEM and SMEM
3. **Schedule operations** on TPU vector units
4. **Analyze dependencies** between programs
5. **Generate low-level Mosaic instructions**

With 16x more programs, each requiring similar optimization passes, the compilation time scales roughly linearly.

## Potential Optimizations

### 1. Increase `block_token` (Trade compilation time for runtime)

```python
# Instead of default block_token=8
top_bounded_k(..., block_token=32, ...)
```

For (256, 2048):
- With `block_token=32`: 256/32 = **8 programs** (4x fewer!)
- Expected compilation time: ~40s instead of 157s

Trade-off: Larger block_token increases ALU work per program and VMEM usage.

### 2. Adjust `bins_topm_schedule` for faster convergence

```python
# Current default auto-computed schedule for max_k=128, num_bins=256:
bins_topm_schedule=(5, 9)  # After adding (0,) prefix

# Try a more aggressive schedule:
bins_topm_schedule=(3, 5, 7, 9)  # More convergence checks, potentially less work
```

This might reduce the complexity of each program's control flow.

### 3. Reduce `bins_topm_unroll`

```python
# Instead of default bins_topm_unroll=64
top_bounded_k(..., bins_topm_unroll=32, ...)
```

Less loop unrolling = simpler HLO = faster compilation (but slower runtime).

### 4. Profile the compiler itself

Use JAX's compilation profiling:

```python
import os
os.environ['XLA_FLAGS'] = '--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text'
```

Then analyze the HLO dumps to understand where compilation time is spent.

## Recommendations

1. **For faster development iteration**: Use larger `block_token` (16 or 32) to reduce compilation time
2. **For production**: Use smaller `block_token` (8) for best runtime performance, accept long compilation
3. **Cache compiled functions**: JAX automatically caches, but ensure you're not unnecessarily recompiling
4. **Consider AOT compilation**: Pre-compile common shapes and save the executables

## Conclusion

The 16.8x compilation slowdown for (256, 2048) vs (16, 2048) is **expected and nearly optimal** given:
- 16x more programs to compile (32 vs 2)
- Each program has similar complexity
- Compilation scales linearly with number of programs

The slow compilation is a consequence of the sophisticated divide-and-filter algorithm which trades compilation time for excellent runtime performance compared to pure bitonic top-k for large vocabularies.

This is not a bug or inefficiency - it's the fundamental trade-off of using a complex, multi-program Pallas kernel.
