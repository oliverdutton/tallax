# Segfault Investigation Results

## Discovery

The segfault seen in pytest was actually a **TypeError** that occurs BEFORE the actual segfault. The real issue is with **descending sorts**, not stable+argsort!

## Test Results

### ✓ PASSING (4/8)
- standard sort
- return_argsort
- stable (no argsort)
- **stable WITH argsort** ← This works fine!

### ✗ FAILING (4/8)
- descending
- descending + argsort
- descending + stable
- descending + stable + argsort

## Root Cause

**Error:** `TypeError: __int__ returned non-int (type DynamicJaxprTracer)`

**Location:** `tallax/_src/bitonic_sort.py:231` in `_compute_is_descending()`

```python
return create_bit_indicator(int(stage), tile_start_offset + int(sort_dim_offset))
                                                            ^^^^^^^^^^^^^^^^^^^^
```

## Why It Fails

1. In `sort.py:102-106`, we compute `sort_dim_offset` as a `SymInt`:
   ```python
   sort_dim_offset = (
       SymInt(pl.program_id(1), 0, pl.num_programs(1)-1) * shape[1] +
       int(descending) * pl.num_programs(1) * shape[1])
   ```

2. When `descending=True`, this offset is passed to `bitonic_sort_arrays()`

3. Inside `bitonic_sort_arrays()`, the code tries: `int(sort_dim_offset)`

4. `SymInt.__int__()` returns the underlying JAX tracer, which is dynamic

5. Python complains: "TypeError: __int__ returned non-int"

## The SymInt Issue

The `SymInt` class was designed to optimize comparisons when bounds allow static evaluation. However, when calling `int()` on a SymInt containing a dynamic tracer, it returns the tracer itself, not an integer.

This works in some contexts but fails when the int() result is used in operations that expect concrete integers.

## Why Ascending Works

When `descending=False`:
- `sort_dim_offset` might be optimized away in certain code paths
- The `_compute_is_descending` function might take different branches
- OR the modulo arithmetic `(sort_dim_offset+i*slice_size) % (2**(stage+1))` evaluates differently

## Next Steps

Need to fix `_compute_is_descending()` to handle SymInt properly without forcing conversion to int when the value is dynamic.
