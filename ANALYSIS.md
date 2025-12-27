# Root Cause Analysis: Descending Sort Failure

## The Problem

When `descending=True`, ALL tests fail with:
```
TypeError: __int__ returned non-int (type DynamicJaxprTracer)
```

## Call Stack

1. `sort.py:102-106` - Computes `sort_dim_offset` as SymInt wrapping `pl.program_id(1)`
   ```python
   sort_dim_offset = (
       SymInt(pl.program_id(1), 0, pl.num_programs(1)-1) * shape[1] +
       int(descending) * pl.num_programs(1) * shape[1])
   ```

2. Passed to `bitonic_sort_arrays(..., sort_dim_offset=sort_dim_offset)`

3. Inside bitonic_sort, line 470:
   ```python
   sort_dim_offset=(sort_dim_offset+i*slice_size) % (2**(stage+1))
   ```
   This creates a new SymInt (modulo operation preserves SymInt)

4. Passed to `_bitonic_sort_substage(..., sort_dim_offset=...)`

5. Then to `_compute_is_descending(..., sort_dim_offset=...)`

6. Line 231 in `_compute_is_descending`:
   ```python
   return create_bit_indicator(int(stage), tile_start_offset + int(sort_dim_offset))
                                                                ^^^^^^^^^^^^^^^^^^^^
   ```

## Why `int(SymInt)` Fails

The `SymInt.__int__()` method (symint.py:168-170):
```python
def __int__(self):
    """Returns tracer value, used as a way to exit SymInt"""
    return self.value
```

This returns `self.value`, which is a JAX `DynamicJaxprTracer`, not a Python `int`.

Python's `int()` builtin expects `__int__()` to return an actual Python integer. When it gets a tracer instead, it raises:
```
TypeError: __int__ returned non-int (type DynamicJaxprTracer)
```

## Why It Works for Ascending

When `descending=False`:
- The sort_dim_offset computation is:
  ```python
  SymInt(pl.program_id(1), ...) * shape[1] + 0 * pl.num_programs(1) * shape[1]
  ```
  Which simplifies to just:
  ```python
  SymInt(pl.program_id(1), ...) * shape[1]
  ```

The key difference is subtle - it might be that the optimizer or SymInt operations handle it differently, OR there's a code path difference in `_compute_is_descending` that we're not seeing.

Actually, looking more carefully - **the same SymInt is created in both cases**. The difference must be in how it's used.

Looking at `_compute_is_descending` again, I see it has optimization paths:

```python
if concrete_and_true((sort_dim_offset % (2**(stage+1))) < 2**stage):
    sort_dim_offset = 0  # Optimization: replace with 0
```

When descending=False, this optimization might trigger more often, replacing the SymInt with a plain 0, avoiding the int() call on the tracer!

## The Fix

Instead of calling `int(sort_dim_offset)`, we should:

1. **Option A**: Check if it's a SymInt and use `.value` directly:
   ```python
   offset_value = sort_dim_offset.value if isinstance(sort_dim_offset, SymInt) else sort_dim_offset
   return create_bit_indicator(int(stage), tile_start_offset + offset_value)
   ```

2. **Option B**: Don't call int() at all - the addition will unwrap it:
   ```python
   # SymInt + int returns a tracer, int + SymInt returns a tracer
   # Both work for create_bit_indicator's index parameter
   if isinstance(sort_dim_offset, SymInt):
       return create_bit_indicator(int(stage), tile_start_offset + sort_dim_offset.value)
   else:
       return create_bit_indicator(int(stage), tile_start_offset + sort_dim_offset)
   ```

3. **Option C**: Make SymInt.__int__() smarter to return a concrete int when possible:
   ```python
   def __int__(self):
       if type(self.value) == int:
           return self.value
       # For tracers, we can't convert to int - raise a better error
       raise TypeError(f"Cannot convert dynamic SymInt to int: {self}")
   ```

**Option B is cleanest** - just access `.value` directly instead of calling `int()`.
