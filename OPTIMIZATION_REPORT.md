# is_descending Optimization Report

## Summary

The `is_descending` computation in bitonic sort can be optimized in many cases from computing a full (8, 128) array to either:
1. **SAME_ALL**: Single constant value for all tiles
2. **CONST_PER_TILE**: One constant value per tile (but differs between tiles)
3. **VARIES**: Must compute full (8, 128) array per tile

## Analysis Results

### Test Case: (8, 128) - Single tile
- **max_local**: 127
- **Pattern**: All stages VARIES except final stage
- Stages 1-6: VARIES (max_local >= 2^stage for stages ≤ 6)
- Stage 7: SAME_ALL (final stage, 2^7 = 128 > max_local)

### Test Case: (8, 256) - First cross-lane
- **num_tiles**: 2
- **max_local**: 247 but chunked in multiples of 16
- **Key insight**: tile_local_offset jumps by 16 (= num_tiles * NUM_SUBLANES)
- Tile 0: offset 0, Tile 1: offset 8
- **Stage 3**: CONST_PER_TILE
  - Tile 0: all offsets 0-7 + (0,16,32,...), bit 3 always 0
  - Tile 1: all offsets 8-15 + (0,16,32,...), bit 3 always 1
  - Optimization possible: `create_bit_indicator(stage, tile_offset)`

### Test Case: (8, 2048) - Many cross-lane
- **num_tiles**: 16
- **max_local**: 1927, chunked in multiples of 128
- **Stages 3-6**: CONST_PER_TILE
  - Tile offsets: 0, 8, 16, 24, 32, ..., 120
  - tile_local_offset increments: 0, 128, 256, ...
  - For low stages, tile_offset dominates and is constant per tile

### Test Case: (128, 256) - Large batch
- **num_tiles**: 32
- **max_local**: 7 (only iota_tile(0) varies!)
- **Stages 3-7**: CONST_PER_TILE
  - With large batch_size, (iota_tile(1) // batch_size) compresses to 0 or 1
  - tile_local_offset ranges only 0-7 within each tile
  - tile_offset dominates for all but the smallest stages

## Optimization Rules

```python
def compute_is_descending_optimized(stage, tile_offset, tile_local_offset, dim1_offset, num_tiles):
    """Compute is_descending with optimizations where possible."""

    max_local = compute_max_tile_local_offset(batch_size, num_tiles)

    # Check if bit varies within tile_local_offset range
    bit_mask = 1 << stage
    if max_local >= bit_mask:
        # Bit might toggle within tile - check if it actually does
        min_in_tile = tile_offset + 0
        max_in_tile = tile_offset + max_local

        if ((min_in_tile >> stage) & 1) != ((max_in_tile >> stage) & 1):
            # VARIES: Must compute full array
            return create_bit_indicator(stage, dim1_offset + tile_offset + tile_local_offset)

    # Bit is constant within each tile - check if same across tiles
    first_tile_bit = create_bit_indicator(stage, dim1_offset + 0)
    all_same = True
    for idx in range(1, num_tiles):
        if create_bit_indicator(stage, dim1_offset + idx * NUM_SUBLANES) != first_tile_bit:
            all_same = False
            break

    if all_same:
        # SAME_ALL: Single value for all tiles
        return first_tile_bit
    else:
        # CONST_PER_TILE: One value per tile
        return create_bit_indicator(stage, dim1_offset + tile_offset)
```

## Cases Where Optimization is NOT Applicable

Based on analysis, **VARIES** case occurs when:

1. **Low stages in single-tile configs**:
   - Example: (8, 128) stages 1-6
   - max_local is large (127) relative to 2^stage
   - Bit toggles within tile_local_offset range

2. **Any stage where bit toggles within tile**:
   - Condition: `((tile_offset + max_local) >> stage) != (tile_offset >> stage)`
   - Example: If tile_offset=0, max_local=127, stage=5
     - Min: 0 (bit 5 = 0)
     - Max: 127 (bit 5 = 1)
     - Bit toggles → must compute full array

## Implementation Impact

For typical use cases:
- **(8, 256)**: 1 of 8 stages optimizable to CONST_PER_TILE
- **(8, 2048)**: 4 of 11 stages optimizable to CONST_PER_TILE
- **(128, 256)**: 5 of 8 stages optimizable to CONST_PER_TILE
- **Final stage**: Always SAME_ALL (can use single bool)

Potential savings:
- CONST_PER_TILE: Avoid (8, 128) computation per tile, use single value
- SAME_ALL: Avoid all computation, use single global value
- Could reduce memory traffic and computation for ~30-50% of stages

## Recommendation

Implement a `compute_is_descending_for_stage` function that:
1. Checks if max_local causes bit to toggle within tile → VARIES
2. Else checks if bit differs across tile offsets → CONST_PER_TILE
3. Else → SAME_ALL

This optimization is especially valuable for:
- Large batch sizes (tile_local_offset range is small)
- Cross-lane stages where tile_offset >> max_local
- Final stages where 2^stage > sort_dim
