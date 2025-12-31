# Recommended Tests to Run on TPU

## Summary of Findings

After correcting my initial misunderstanding (grid size does NOT affect compilation), here's what we know:

**Observed Compilation Times:**
- (16, 2048): 16.49s total (7.13s jaxpr + 6.61s lowering + 2.75s compilation)
- (256, 2048): 254.36s total (96.97s jaxpr + 87.17s lowering + 70.22s compilation)
- **Ratio: ~15.4x total, with jaxpr/lowering ~13x and compilation ~25x**

**Key Changes Between Shapes:**
1. **Scratch buffer sizes**: 16x larger (589,824 vs 36,864 elements in VMEM)
2. **Input/output arrays**: 16x more tokens to process
3. **Same kernel body**: The Pallas kernel compiles once regardless of grid size

## Hypothesis

The slowdown is caused by:
1. **Larger scratch buffers** requiring more complex memory management in Mosaic
2. **Unrolled loops** (`bins_topm_unroll=64`) creating more jaxpr equations during tracing
3. **XLA optimization passes** scaling with buffer sizes

## Tests to Run on TPU

### Test 1: Measure Lowering vs Compilation Time

Run your existing timing script and separate out the stages:

```python
import time

# Measure jaxpr creation
t0 = time.perf_counter()
jaxpr = jax.make_jaxpr(top_bounded_k)(logits, k, ...)
t_jaxpr = time.perf_counter() - t0

# Measure lowering
t0 = time.perf_counter()
lowered = jax.jit(top_bounded_k).lower(logits, k, ...)
t_lower = time.perf_counter() - t0

# Measure compilation
t0 = time.perf_counter()
compiled = lowered.compile()
t_compile = time.perf_counter() - t0

print(f"Jaxpr: {t_jaxpr:.2f}s, Lower: {t_lower:.2f}s, Compile: {t_compile:.2f}s")
```

**Expected**: All three stages should scale similarly (~13-16x).

### Test 2: Reduce `bins_topm_unroll`

The default is `bins_topm_unroll=64`. Try smaller values:

```python
# Test with different unroll factors
for unroll in [16, 32, 64]:
    t0 = time.perf_counter()
    result = top_bounded_k(
        logits,
        k,
        max_k=128,
        bins_topm_unroll=unroll,  # <-- Change this
        ...
    )
    t1 = time.perf_counter()
    print(f"unroll={unroll}: {t1-t0:.2f}s")
```

**Expected**: Smaller unroll = faster jaxpr creation but possibly slower runtime.

### Test 3: Test Intermediate Batch Sizes

Plot compilation time vs num_tokens to see the scaling pattern:

```python
batch_sizes = [16, 32, 64, 128, 256]
compile_times = []

for num_tokens in batch_sizes:
    shape = (num_tokens, 2048)
    logits = jax.random.normal(key, shape).astype(jnp.bfloat16)
    k = jnp.full((num_tokens,), 128, dtype=jnp.int32)

    t0 = time.perf_counter()
    result = top_bounded_k(logits, k, max_k=128, ...)
    t1 = time.perf_counter()

    compile_times.append(t1 - t0)
    print(f"{num_tokens}: {t1-t0:.2f}s")

# Plot or analyze scaling
import matplotlib.pyplot as plt
plt.plot(batch_sizes, compile_times, 'o-')
plt.xlabel('Batch Size')
plt.ylabel('Compilation Time (s)')
plt.title('Compilation Time Scaling')
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.savefig('compilation_scaling.png')
```

**Expected**: If it's linear (O(n)), the log-log plot should have slope ~1.

### Test 4: Compare with Bitonic Top-K

For the same shapes, compare `top_bounded_k` vs `bitonic_top_k`:

```python
shapes = [(16, 2048), (256, 2048)]

for shape in shapes:
    logits = jax.random.normal(key, shape).astype(jnp.bfloat16)

    # Test bitonic_top_k
    t0 = time.perf_counter()
    _ = bitonic_top_k(logits, k=128)
    t_bitonic = time.perf_counter() - t0

    # Test top_bounded_k
    t0 = time.perf_counter()
    _ = top_bounded_k(logits, k=..., max_k=128, ...)
    t_bounded = time.perf_counter() - t0

    print(f"{shape}:")
    print(f"  bitonic: {t_bitonic:.2f}s")
    print(f"  bounded: {t_bounded:.2f}s")
    print(f"  ratio: {t_bounded/t_bitonic:.2f}x")
```

**Expected**: From your data:
- (16, 2048): bitonic 3.04s, bounded ~6.6s (lower only)
- (256, 2048): bitonic 13.11s (4.3x), bounded ~87s (13.2x)

This confirms `top_bounded_k` scales worse than `bitonic_top_k`.

### Test 5: Dump and Analyze HLO

Dump the HLO to see if the IR size itself is 16x larger:

```python
import os
os.environ['XLA_FLAGS'] = '--xla_dump_to=/tmp/xla_dump_16 --xla_dump_hlo_as_text'

# Compile for (16, 2048)
logits_16 = jax.random.normal(key, (16, 2048)).astype(jnp.bfloat16)
_ = top_bounded_k(logits_16, k=..., max_k=128, ...)

os.environ['XLA_FLAGS'] = '--xla_dump_to=/tmp/xla_dump_256 --xla_dump_hlo_as_text'

# Compile for (256, 2048)
logits_256 = jax.random.normal(key, (256, 2048)).astype(jnp.bfloat16)
_ = top_bounded_k(logits_256, k=..., max_k=128, ...)
```

Then compare the HLO files:

```bash
# Count lines in each HLO dump
wc -l /tmp/xla_dump_16/*.hlo.txt
wc -l /tmp/xla_dump_256/*.hlo.txt

# Look for differences in buffer allocations
grep -i "allocate\|buffer" /tmp/xla_dump_16/*.hlo.txt | wc -l
grep -i "allocate\|buffer" /tmp/xla_dump_256/*.hlo.txt | wc -l
```

**Expected**: If HLO size is proportional to buffer size, you'll see ~16x more lines in the (256, 2048) version.

### Test 6: Vary `block_token`

Test if using larger `block_token` helps (fewer programs, but same scratch buffer size):

```python
for block_token in [8, 16, 32]:
    logits = jax.random.normal(key, (256, 2048)).astype(jnp.bfloat16)
    k = jnp.full((256,), 128, dtype=jnp.int32)

    num_programs = (256 + block_token - 1) // block_token

    t0 = time.perf_counter()
    result = top_bounded_k(
        logits,
        k,
        max_k=128,
        block_token=block_token,  # <-- Change this
        ...
    )
    t1 = time.perf_counter()

    print(f"block_token={block_token} (num_programs={num_programs}): {t1-t0:.2f}s")
```

**Expected**: Compilation time should be similar since:
- Scratch buffers stay `(256, 2304)` regardless
- Only grid size changes (32 → 16 → 8 programs)
- But if grid size DOES matter (contrary to docs), this will show it!

## What to Look For

1. **Linear scaling (O(n))**: Expected and probably unavoidable
2. **Superlinear scaling (O(n²) or worse)**: Would indicate a compiler bug or inefficiency
3. **Constant overhead**: A fixed compilation cost independent of batch size
4. **Unroll impact**: If reducing `bins_topm_unroll` significantly speeds up jaxpr creation, that's the bottleneck

## My Prediction

Based on the data, I expect:
1. **~13-16x linear scaling** is fundamental - larger buffers require proportionally more work
2. **Reducing `bins_topm_unroll`** might help jaxpr creation by 2-3x
3. **block_token won't matter** for compilation (but might affect runtime)
4. **HLO size will be ~16x larger** due to buffer allocations

If any of these predictions are wrong, that would reveal something interesting about the compiler!

## Next Steps After Tests

If you confirm linear scaling:
- **Accept it**: For 16x larger inputs, 13-16x compilation time is reasonable
- **Use AOT compilation**: Pre-compile for common shapes and cache
- **Increase block_token**: Process more tokens per program to reduce total scratch size (trade compile time for runtime)

If you find superlinear scaling:
- **File a JAX bug**: O(n²) or worse would be a compiler inefficiency
- **Investigate specific passes**: Use XLA profiling to find the slow optimization pass
