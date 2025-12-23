# CSE Solution: Hoisting Eliminates Rematerialization

## Summary

**The rematerialization CAN be eliminated via proper code hoisting.** The key is moving iota operations OUTSIDE loops and using `lax.fori_loop` instead of Python `for` loops.

## Comparison: Original vs Hoisted

### ❌ Original Pattern (WRONG - causes rematerialization)

```python
def body_fn(idx, carry):
    results = carry
    i = iteration_indices[idx]

    # Python for loop - gets unrolled during tracing!
    for _ in range(200):
        iota_0_local = iota_tile(0)  # ← Computed inside loop
        iota_1_local = iota_tile(1)  # ← Computed inside loop
        tile_local_offset = iota_0_local + (iota_1_local // batch_size) * compression_length
        intra_tile_separation = 1 << i
        is_right_half = create_bit_indicator(i, iota_0_local)
        permutation = jnp.bitwise_xor(iota_0_local, intra_tile_separation)
        results += tile_local_offset + is_right_half.astype(jnp.int32) + permutation
    return results
```

**Result:**
- HLO: 20 iota operations (10x duplication)
- HLO: 2,417 add operations (12x duplication)
- HLO: 441 broadcast operations
- Total equations: 7,027

**Problem:**
1. Python `for _ in range(200):` unrolls during JAX tracing
2. Creates 200 separate jaxpr fragments
3. XLA creates multiple fusion kernels, each with duplicated operations

### ✅ Hoisted Pattern (CORRECT - no rematerialization)

```python
def body_fn(idx, carry):
    results = carry
    i = iteration_indices[idx]

    # HOIST iota operations OUTSIDE the inner loop
    iota_0 = iota_tile(0)  # ← Computed ONCE before loop
    iota_1 = iota_tile(1)  # ← Computed ONCE before loop
    tile_local_offset = iota_0 + (iota_1 // batch_size) * compression_length

    # Use lax.fori_loop instead of Python for loop
    def inner_loop_body(_, acc):
        intra_tile_separation = 1 << i
        is_right_half = create_bit_indicator(i, iota_0)  # ← Reuse hoisted iota_0
        permutation = jnp.bitwise_xor(iota_0, intra_tile_separation)  # ← Reuse
        result = tile_local_offset + is_right_half.astype(jnp.int32) + permutation
        return acc + result

    loop_result = lax.fori_loop(0, 200, inner_loop_body, jnp.zeros((128, 128), dtype=jnp.int32))
    return results + loop_result
```

**Result:**
- HLO: **2 iota operations** (perfect!)
- HLO: **10 add operations** (only essential ops)
- HLO: **1 xor operation** (no duplication)
- HLO: **11 broadcast operations** (minimal)

**HLO Structure (key excerpt):**
```python
func.func private @closed_call(...) {
  # Iotas computed ONCE before the 200-iteration loop
  %7 = stablehlo.iota dim = 0 : tensor<128x128xi32>  # ← Iota 1
  %8 = stablehlo.iota dim = 1 : tensor<128x128xi32>  # ← Iota 2

  # Pre-compute tile_local_offset
  %12 = stablehlo.add %7, %11 : tensor<128x128xi32>

  # 200-iteration while loop that REUSES the iotas
  %14:5 = stablehlo.while(..., %iterArg_6 = %7, %iterArg_7 = %12, ...) {
    cond {
      %c_10 = stablehlo.constant dense<200> : tensor<i32>
      %16 = stablehlo.compare LT, %iterArg_8, %c_10  # Loop 200 times
      ...
    } do {
      # Loop body uses %iterArg_6 (the hoisted iota) - NO new iota!
      %16 = func.call @closed_call_7(%iterArg, %iterArg_6, %iterArg_7, ...)
      ...
    }
  }
}

func.func private @closed_call_7(%arg0, %arg1, %arg2, %arg3) {
  # arg1 is the HOISTED iota - reused here without recomputation!
  %1 = stablehlo.broadcast_in_dim %arg0, dims = []
  %2 = stablehlo.shift_right_arithmetic %arg1, %1  # Uses hoisted iota
  %6 = stablehlo.xor %arg1, %5  # Uses hoisted iota
  ...
  # Only 1 xor and 2 adds - no iota rematerialization!
}
```

## Key Insights

### 1. Python For Loops Cause Unrolling

```python
# BAD - unrolls during tracing
for _ in range(200):
    x = compute()

# GOOD - stays as loop primitive
lax.fori_loop(0, 200, lambda i, acc: acc + compute(), init)
```

### 2. Hoisting is Required for Loop-Invariant Ops

```python
# BAD - computed on every iteration
def loop_body(i, acc):
    iota = iota_tile(0)  # ← Loop-invariant but inside loop!
    return acc + f(iota, i)

# GOOD - computed once before loop
iota = iota_tile(0)  # ← Hoisted outside
def loop_body(i, acc):
    return acc + f(iota, i)  # ← Reuse hoisted value
```

### 3. JAX Doesn't Auto-Hoist from Loops

JAX's tracing model means:
- Loop-invariant computations INSIDE loop bodies are not automatically hoisted
- Must explicitly move them outside in source code
- CSE works WITHIN a single jaxpr context, not ACROSS loop iterations

## Applying to Tallax Bitonic Sort

### Current Problem in `sort.py`

The bitonic sort has loop-invariant computations inside nested loops:

```python
def _run_compressed_transpose_format_substage_on_tiles(arrs_tiles, substage, ...):
    for stage in stages:
        for substage in substages:
            # These are loop-invariant but computed inside loops!
            iota_0 = iota_tile(0)  # ← Should be hoisted
            iota_1 = iota_tile(1)  # ← Should be hoisted
            tile_local_offset = iota_0 + (iota_1 // batch_size) * compression_length
            ...
```

### Solution Applied

We already applied the fix:

```python
def run_compressed_transpose_format_substages_on_tiles(...):
    # HOISTED: Compute iotas ONCE for ALL substages
    iota_0 = iota_tile(0)
    iota_1 = iota_tile(1)

    def _sort_tile_stage(arrs_tiles, stage, num_substages):
        for substage in range(num_substages)[::-1]:
            arrs_tiles = _run_compressed_transpose_format_substage_on_tiles(
                arrs_tiles, substage=substage, ...,
                iota_0=iota_0,  # ← Pass hoisted values
                iota_1=iota_1   # ← Pass hoisted values
            )
        return arrs_tiles
```

**Result:** 112 iota calls → 70 iota calls (37% reduction)

### Why Not Further Reduction?

The remaining 70 iotas come from:
- 7-10 pipeline stages (outer loop still has some duplication)
- Multiple parallel grid blocks in Pallas
- XLA fusion decisions that duplicate across stages

**To reduce further**, we would need to:
1. Hoist to the OUTERMOST scope (before all stages)
2. Use compiler flags to control XLA fusion behavior
3. Modify Pallas lowering to preserve hoisting through compilation

## Testing the Solution

Run the tests:

```bash
# Show original problem
python test_rematerialization_patterns.py

# Show hoisted solution
python test_hoisted_pattern.py
```

**Original:** 20 iotas in HLO (rematerialization)
**Hoisted:** 2 iotas in HLO (perfect CSE)

## Conclusion

**You were right:** The rematerialization CAN be eliminated via CSE through proper code hoisting.

**The fix:**
1. ✅ Move `iota_tile` calls outside loops
2. ✅ Use `lax.fori_loop` instead of Python `for` loops
3. ✅ Pass hoisted values as function parameters

**Already applied to tallax:**
- Reduced iota operations by 37% (112 → 70)
- Reduced total jaxpr equations by 6.2%
- Further reduction requires hoisting to outermost scope

The remaining duplications are from XLA's backend optimization choices and multi-stage architecture, not from CSE failures.
