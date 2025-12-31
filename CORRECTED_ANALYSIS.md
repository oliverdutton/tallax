# Corrected Analysis: Why (256, 2048) Compiles Slower Than (16, 2048)

## Correction: Grid Size Does NOT Affect Compilation

**I was wrong in my initial analysis.** As per Pallas documentation, the grid parameter works like a for loop:

```python
pl.pallas_call(kernel, grid=(n,))(...)
# is equivalent to:
for i in range(n):
    kernel(...)
```

**The kernel body is compiled ONCE, then executed n times.** So having 32 programs vs 2 programs does NOT directly cause 16x compilation time.

## Observed Timings (from your output)

### (16, 2048):
- Jaxpr creation: 7.13s
- Lowering: 6.61s
- Compilation: 2.75s
- **Total: 16.49s**

### (256, 2048):
- Jaxpr creation: 96.97s (13.6x slower!)
- Lowering: 87.17s (13.2x slower!)
- Compilation: 70.22s (25.5x slower!)
- **Total: 254.36s (15.4x slower overall)**

## What ACTUALLY Changes?

Between (16, 2048) and (256, 2048), here's what changes in `top_bounded_k`:

### 1. **Scratch Buffer Shapes** (16x larger)

```python
# For (16, 2048):
VMEM scratch: (16, 2304) = 36,864 elements
SMEM arrays: (16,) for max_depth and cutoff_vals

# For (256, 2048):
VMEM scratch: (256, 2304) = 589,824 elements (16x larger!)
SMEM arrays: (256,) for max_depth and cutoff_vals
```

### 2. **Input/Output Array Shapes**

```python
# Inputs:
logits: (16, 2048) → (256, 2048)
k: (16,) → (256,)

# Outputs:
topk_vals: (16, 128) → (256, 128)
topk_idxs: (16, 128) → (256, 128)
depths: (16,) → (256,)
cutoff_vals: (16,) → (256,)
```

### 3. **Index Calculations**

The kernel uses indexing like:
```python
max_depth_ref[pid * block_token + i] = ...
```

With larger arrays, these index calculations and memory accesses might be more complex for the compiler to optimize.

## Hypotheses for the Slowdown

### Hypothesis 1: Jaxpr Creation Scales with Array Sizes

The **jaxpr creation** is already 13.6x slower. This happens during Python-level tracing, before any TPU-specific lowering.

Possible causes:
- **Unrolled loops**: The code uses `unrolled_fori_loop(length, body, init, unroll=64)`, which creates 64 Python-level iterations of the body during tracing
- **List comprehensions**: There are many list comprehensions like:
  ```python
  bins_topk_vals=[
      bins_topm_vals_ref[token_slice, pl.dslice(i * num_bins, num_bins)]
      for i in range(m)
  ]
  ```
- **Array operations**: JAX needs to trace through all the array operations with the larger shapes

### Hypothesis 2: TPU Mosaic Lowering Scales with Buffer Sizes

The **lowering** stage (converting to TPU-specific Mosaic IR) is also 13.2x slower.

Possible causes:
- **Memory layout**: The Mosaic compiler needs to figure out how to lay out 16x larger buffers in VMEM/SMEM
- **DMA operations**: More complex DMA transfers between HBM ↔ VMEM ↔ SMEM
- **Index optimization**: More complex index arithmetic for larger arrays
- **Scratch allocation**: The VMEM allocation strategy might be more complex with larger buffers

### Hypothesis 3: XLA Compilation Scales Super linearly

The **compilation** stage (XLA optimizing the HLO) is 25.5x slower - worse than the input size ratio!

Possible causes:
- **Optimization passes**: XLA runs multiple optimization passes (constant folding, CSE, fusion, etc.) that might scale poorly with program size
- **Register allocation**: Larger scratch buffers require more sophisticated register allocation
- **Memory bandwidth optimization**: The compiler tries to optimize memory access patterns, which gets more complex with larger buffers

## Testing the Hypotheses

### Test 1: Reduce `bins_topm_unroll`

The default `bins_topm_unroll=64` creates 64 unrolled iterations. Try reducing it:

```python
top_bounded_k(..., bins_topm_unroll=32, ...)  # or even 16
```

**Expected**: If unrolling is the issue, this should speed up jaxpr creation.

### Test 2: Increase `block_token`

With larger `block_token`, the scratch buffers stay the same size but process more tokens per block:

```python
top_bounded_k(..., block_token=16, ...)  # Instead of 8
```

For (256, 2048) with `block_token=16`:
- VMEM scratch: (256, 2304) - **same size!**
- num_programs: 16 instead of 32

**Expected**: If buffer size is the issue, this won't help (buffer still 256 tokens). But worth testing.

### Test 3: Smaller Input Shapes

Test intermediate sizes to see if the scaling is linear or superlinear:

```python
shapes_to_test = [
    (16, 2048),
    (32, 2048),   # 2x
    (64, 2048),   # 4x
    (128, 2048),  # 8x
    (256, 2048),  # 16x
]
```

Plot compilation time vs num_tokens to see if it's O(n), O(n log n), or O(n²).

## Key Question: Why Does Buffer Size Matter?

Since the kernel body compiles once regardless of grid size, why do larger buffers cause slower compilation?

**Possible Answer**: The compiler still needs to:
1. **Allocate the buffers**: Larger allocations require more complex memory management
2. **Generate DMA code**: Transfers between HBM and VMEM scale with buffer size
3. **Optimize index arithmetic**: `pid * block_token + i` indexing into larger arrays
4. **Analyze data dependencies**: More data means more potential dependencies to track

Even though the *kernel logic* is the same, the *data layout and movement* is different, and the compiler needs to optimize that.

## Next Steps

1. **Profile with XLA_FLAGS**: Dump HLO to see if the HLO size itself is 16x larger
   ```bash
   export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text"
   ```

2. **Test reduced unroll**: Try `bins_topm_unroll=16` or `32`

3. **Test intermediate sizes**: Plot compilation time vs num_tokens

4. **Check Mosaic IR**: The TPU Mosaic IR might reveal what's actually different

## Conclusion

The compilation slowdown is NOT due to grid size (which was my initial mistake). It's likely due to:

1. **Larger scratch buffers** requiring more complex memory management
2. **Unrolled loops** creating more jaxpr equations during tracing
3. **XLA optimization passes** scaling poorly with buffer sizes

The 13-16x slowdown for 16x larger inputs suggests roughly **linear scaling**, which might be unavoidable if the compiler needs to process proportionally more data.
