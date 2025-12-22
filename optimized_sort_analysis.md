# Rematerialization Analysis and Fix

## Problem Identified

The Pallas jaxpr shows significant rematerialization:
- **Total equations**: 8,824
- **Unique equations**: 4,902
- **Duplicate operations**: 3,922 (44% redundancy!)
- **Potential rematerializations**: 109 distinct operations

## Most Duplicated Operations

1. **`iota()` - 112 times**: Creating index arrays repeatedly
2. **`lt`, `add`, `select_n`, `reshape`, `gather` - 336 times each**: Bitonic comparison logic
3. **`convert_element_type`, `div`, `sign`, `rem` - 35 times each**: Index calculations

## Root Causes

### 1. iota_tile() Rematerialization

In `utils.py:268-270`:
```python
def iota_tile(dim):
    """Create iota array with tile shape."""
    return lax.broadcasted_iota(jnp.int32, (NUM_SUBLANES, NUM_LANES), dim)
```

This is called from `create_bit_indicator` (line 279) every time without caching:
```python
if index is None:
    index = iota_tile(1)  # Creates new array each time!
```

### 2. Loop Body Rematerialization

In `_run_compressed_transpose_format_substage_on_tiles` (lines 117-178), the function computes:
- `iota_tile(0)` and `iota_tile(1)` multiple times (lines 134, 152, 156)
- `create_bit_indicator` repeatedly with the same arguments
- Permutations and comparisons that could be hoisted

### 3. Pipeline Loop Inefficiency

In `_run_array_substage_on_hbm_refs` (lines 596-704):
- `perform_dma` creates new index calculations for each iteration
- `compute` recalculates bit indicators and indices

## Proposed Fixes

### Fix 1: Cache iota_tile Results (Easy, High Impact)

Create cached versions at module or function scope:
```python
# Pre-compute common iota tiles
_IOTA_TILE_0 = None
_IOTA_TILE_1 = None

def get_iota_tile(dim):
    global _IOTA_TILE_0, _IOTA_TILE_1
    if dim == 0:
        if _IOTA_TILE_0 is None:
            _IOTA_TILE_0 = lax.broadcasted_iota(jnp.int32, (NUM_SUBLANES, NUM_LANES), 0)
        return _IOTA_TILE_0
    elif dim == 1:
        if _IOTA_TILE_1 is None:
            _IOTA_TILE_1 = lax.broadcasted_iota(jnp.int32, (NUM_SUBLANES, NUM_LANES), 1)
        return _IOTA_TILE_1
    return lax.broadcasted_iota(jnp.int32, (NUM_SUBLANES, NUM_LANES), dim)
```

### Fix 2: Hoist Loop-Invariant Computations

In `_run_compressed_transpose_format_substage_on_tiles`:
```python
# Pre-compute these once before loops
tile_local_offset = iota_tile(0) + (iota_tile(1) // batch_size) * num_tiles * NUM_SUBLANES
iota_0 = iota_tile(0)

# Then reuse in loops
```

### Fix 3: Jaxpr-Level CSE Pass

Implement a proper CSE pass that:
1. Identifies duplicate computations in the jaxpr
2. Hoists them out of loops
3. Reuses computed values

This would require integration with JAX's compilation pipeline.

### Fix 4: Code Restructuring (Current Best Approach)

Since jaxpr-level optimization is complex, restructure the Python code to:
1. Pre-compute all loop-invariant values before loops
2. Pass them as arguments to avoid recomputation
3. Use let-bindings to explicitly share computations

## Recommended Implementation Order

1. **Fix 1** (iota_tile caching) - Quick win, ~112 operations saved
2. **Fix 4** (code restructuring) - Hoist computations in critical loops
3. **Fix 3** (CSE pass) - If needed after measuring impact of 1 & 2

## Expected Impact

Conservative estimate: Reducing 3,922 duplicate operations by 50-70% would:
- Reduce jaxpr size from 8,824 to ~5,000-6,000 equations
- Improve compilation time
- Reduce runtime overhead from redundant calculations
- Lower memory pressure

## Testing Plan

1. Run `analyze_rematerialization.py` before and after changes
2. Compare jaxpr equation counts
3. Benchmark actual sort performance
4. Verify correctness with existing tests
