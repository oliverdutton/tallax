# is_descending Optimization Report

## Summary

The `is_descending` computation in bitonic sort can be optimized based on the stage and configuration. Following the pattern from `sort.py`, we have three optimization cases:

1. **SAME_2D_FOR_ALL_TILES**: When `stage < log2(NUM_SUBLANES)` (stage < 3)
   - Returns a (8, 128) array based on `tile_local_offset` only
   - This array is the same for all tiles, so can be computed once and reused

2. **SCALAR_PER_TILE**: When `stage < log2(num_tiles * NUM_SUBLANES)` (stage < max_substage)
   - Returns a single scalar boolean per tile
   - Different tiles may have different values

3. **FULL_ARRAY**: For larger stages
   - Returns full (8, 128) array per tile
   - Values vary within each tile

## Optimization Rules (from sort.py)

```python
def compute_is_descending(idx):
    tile_offset = idx * NUM_SUBLANES
    is_desc = create_bit_indicator(stage, dim1_offset + tile_offset + tile_local_offset)

    if type(stage) == int:
        if stage < log2(NUM_SUBLANES):
            # every tile has same (8,128) value
            return create_bit_indicator(stage, tile_local_offset + dim1_offset)
        elif stage < log2(num_tiles * NUM_SUBLANES):
            # value constant across tile (scalar per tile)
            return create_bit_indicator(stage, tile_offset + dim1_offset)

    return is_desc
```

## Analysis Results

### Test Case: (8, 128) - Single tile
- `num_tiles = 1`, `max_substage = log2(8) = 3`
- Stages 1-2: **SAME_2D_FOR_ALL_TILES** (stage < 3)
- Stage 3: **SAME_2D_FOR_ALL_TILES** (stage < 3)
- Stages 4-6: **FULL_ARRAY** (stage >= max_substage)
- Stage 7: **FULL_ARRAY** (final stage, but varies within tile)

### Test Case: (8, 256) - First cross-lane
- `num_tiles = 2`, `max_substage = log2(16) = 4`
- Stages 1-2: **SAME_2D_FOR_ALL_TILES** (stage < 3)
- Stages 3: **SCALAR_PER_TILE** (3 < 4)
- Stages 4-7: **FULL_ARRAY** (stage >= max_substage)
- Stage 8: **FULL_ARRAY** (final stage)

### Test Case: (8, 2048) - Many cross-lane
- `num_tiles = 16`, `max_substage = log2(128) = 7`
- Stages 1-2: **SAME_2D_FOR_ALL_TILES** (stage < 3)
- Stages 3-6: **SCALAR_PER_TILE** (3 <= stage < 7)
- Stages 7-10: **FULL_ARRAY** (stage >= max_substage)
- Stage 11: **FULL_ARRAY** (final stage)

### Test Case: (128, 256) - Large batch
- `num_tiles = 32`, `max_substage = log2(256) = 8`
- Stages 1-2: **SAME_2D_FOR_ALL_TILES** (stage < 3)
- Stages 3-7: **SCALAR_PER_TILE** (3 <= stage < 8)
- Stage 8: **FULL_ARRAY** (stage >= max_substage)

## Implementation Impact

The simple rules based on stage thresholds provide effective optimization:

### For typical use cases:
- **(8, 256)**:
  - 3 stages as SAME_2D_FOR_ALL_TILES (stages 1-3)
  - 0 stages as SCALAR_PER_TILE (none fit the window)
  - Actually stage 3 is SCALAR_PER_TILE, so 3 of 8 stages optimized

- **(8, 2048)**:
  - 2 stages as SAME_2D_FOR_ALL_TILES (stages 1-2)
  - 4 stages as SCALAR_PER_TILE (stages 3-6)
  - 6 of 11 stages optimized (~54%)

- **(128, 256)**:
  - 2 stages as SAME_2D_FOR_ALL_TILES (stages 1-2)
  - 5 stages as SCALAR_PER_TILE (stages 3-7)
  - 7 of 8 stages optimized (~87%)

### Benefits:
- **SAME_2D_FOR_ALL_TILES**: Compute once, reuse for all tiles
- **SCALAR_PER_TILE**: Use single bool instead of (8, 128) array
- **Reduced memory traffic**: Fewer array operations
- **Simpler code**: Clean threshold-based rules

### When FULL_ARRAY is required:
- Cross-lane substages where `stage >= log2(num_tiles * NUM_SUBLANES)`
- These are the stages where tile_local_offset contribution causes within-tile variation
- Example: (8, 2048) stages 7-11 require full computation

## Implementation

Implemented in `tallax/_src/bitonic_topk.py`:
```python
def _compute_is_descending_for_tile(stage, tile_idx, batch_size, num_tiles,
                                     dim1_offset, tile_local_offset):
    """Compute is_descending for a tile with optimizations."""
    tile_offset = tile_idx * NUM_SUBLANES
    is_desc = create_bit_indicator(stage, dim1_offset + tile_offset + tile_local_offset)

    if type(stage) == int:
        if stage < log2(NUM_SUBLANES):
            # every tile has same (8,128) value
            return create_bit_indicator(stage, tile_local_offset + dim1_offset)
        elif stage < log2(num_tiles * NUM_SUBLANES):
            # value constant across tile (scalar per tile)
            return create_bit_indicator(stage, tile_offset + dim1_offset)

    return is_desc
```

This matches the proven pattern from `sort.py` and provides significant performance benefits for most configurations.
