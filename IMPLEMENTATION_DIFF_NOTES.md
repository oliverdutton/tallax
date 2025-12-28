# Implementation Differences: Main vs Simplify_Sort

## Main Branch: `_bitonic_sort_arrays`

**Input:** `arrs_tiles` - list-of-arrays (e.g., [array(128,128)])
**Output:** `arrs_tiles` - list-of-lists (e.g., [[tile1, tile2]])

**Key characteristics:**
1. Directly operates on array tiles
2. Explicitly passes `compression_length` parameter
3. Two-phase loop structure:
   - Phase 1: Stages 1 to `stage_unroll`
   - Phase 2: Stages `stage_unroll+1` to `num_stages`
4. Uses modulo arithmetic: `(sort_dim_offset+i*slice_size) % (2**(stage+1))`
5. Returns list-of-lists tile structure directly

## Simplify_Sort Branch: `_bitonic_sort_substages_maybe_refs`

**Input:** `inputs` - can be arrays OR refs
**Output:** When arrays: `[process_block(0), ...]` - list of outer_out_tiles per grid block

**Key characteristics:**
1. Unifies array and ref handling
2. Computes `compression_length` internally from input structure
3. Single schedule of all (substage, stage) pairs
4. Returns wrapped structure: list with one element per grid block
5. When `grid_size=1`, returns `[outer_out_tiles]`

## Critical Differences

### Return Structure
- **Main:** Returns tiles directly as list-of-lists
- **Simplify:** Returns list wrapping the tiles (needs `outputs[0]` extraction)

### sort_dim_offset Handling
- **Main:** Uses modulo `% (2**(stage+1))` for each slice
- **Simplify:** Uses `SymInt` with tile_offset computation
  - `tile_offset = sort_dim_offset + SymInt(outer_i, ...) * outer_size + SymInt(inner_i) * inner_size`

### Missing in Simplify_Sort
1. **NO modulo arithmetic** on sort_dim_offset during substage iteration
2. **NO `concat_threshold` parameter** passed to `bitonic_sort_substage` in early stages
3. **NO `compression_length` parameter** - computed internally instead

## Suspected Bug

The main branch carefully computes different `sort_dim_offset` for each slice:
```python
sort_dim_offset=(sort_dim_offset+i*slice_size) % (2**(stage+1))
```

But simplify_sort computes:
```python
tile_offset = sort_dim_offset + SymInt(outer_i, 0, grid_size-1) * outer_size + SymInt(inner_i) * inner_size
```

For stage=8 (256 elements), the modulo wrapping at `2**(stage+1) = 512` is important for bitonic sort correctness.
**Without the modulo, the sort_dim_offset can exceed the pattern period, causing incorrect comparisons!**

This would explain why (8,256) fails but (8,128) passes:
- (8,128): num_stages=7, 2^8 = 256 > 128 ✓
- (8,256): num_stages=8, 2^9 = 512 > 256 but tile_offset can grow beyond pattern!
