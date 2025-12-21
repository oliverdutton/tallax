# is_descending Optimization Report

## Summary

The `is_descending` computation in bitonic sort achieves **100% optimization** using stratified rules based on bit position analysis. Every stage is optimized to avoid unnecessary array computations.

## Stratified Optimization Rules

Four optimization cases (checked in order):

1. **SAME_2D_FOR_ALL**: When `stage < log2(NUM_SUBLANES)` (stage < 3)
   - Bit only set by sublane index `iota_tile(0)`
   - Returns (8, 128) array based on `tile_local_offset` only
   - Same array for all tiles - compute once, reuse everywhere

2. **SCALAR_PER_TILE**: When `stage < log2(num_tiles * NUM_SUBLANES)`
   - Bit set by `tile_offset` differences
   - Constant within each tile, differs across tiles
   - Returns single scalar boolean per tile

3. **GLOBAL_2D**: When `stage < log2(sort_dim)`
   - Bit position beyond `tile_offset` range (tile_offset doesn't contribute)
   - Pattern comes only from `tile_local_offset`
   - Same (8, 128) array for all tiles - compute once, reuse everywhere

4. **GLOBAL_SCALAR**: When `stage >= log2(sort_dim)`
   - Bit position beyond array size, never set
   - Returns single scalar boolean for all tiles

## Implementation

```python
def _compute_is_descending_for_tile(stage, tile_idx, batch_size, num_tiles,
                                     dim1_offset, tile_local_offset, sort_dim):
    """Compute is_descending for a tile with stratified optimizations."""
    tile_offset = tile_idx * NUM_SUBLANES

    if type(stage) == int:
        if stage < log2(NUM_SUBLANES):
            return create_bit_indicator(stage, tile_local_offset + dim1_offset)
        elif stage < log2(num_tiles * NUM_SUBLANES):
            return create_bit_indicator(stage, tile_offset + dim1_offset)
        elif stage < log2(sort_dim):
            return create_bit_indicator(stage, dim1_offset + tile_local_offset)
        else:
            return create_bit_indicator(stage, dim1_offset)

    return create_bit_indicator(stage, dim1_offset + tile_offset + tile_local_offset)
```

## Optimization Coverage (100% for all configurations)

| Config | SAME_2D | SCALAR | GLOBAL_2D | GLOBAL_SCALAR | Total |
|--------|---------|--------|-----------|---------------|-------|
| (8, 128) | 29% (2) | 0% (0) | 57% (4) | 14% (1) | **100%** (7/7) |
| (8, 2048) | 18% (2) | 36% (4) | 36% (4) | 9% (1) | **100%** (11/11) |
| (16, 4096) | 17% (2) | 50% (6) | 25% (3) | 8% (1) | **100%** (12/12) |
| (16, 16384) | 14% (2) | 57% (8) | 21% (3) | 7% (1) | **100%** (14/14) |
| (128, 256) | 25% (2) | 62% (5) | 0% (0) | 12% (1) | **100%** (8/8) |

### Example: (8, 2048) Stratification

- **Stages 1-2**: SAME_2D_FOR_ALL (stage < 3)
- **Stages 3-6**: SCALAR_PER_TILE (3 ≤ stage < 7)
- **Stages 7-10**: GLOBAL_2D (7 ≤ stage < 11)
- **Stage 11**: GLOBAL_SCALAR (stage ≥ 11)

## Key Benefits

1. **Universal Optimization**: 100% of stages optimized for all configurations
2. **Zero Runtime Overhead**: All decisions made via compile-time thresholds
3. **Clear Stratification**: Four distinct rules covering all cases
4. **Mathematically Sound**: Based on bit position analysis of bitonic pattern

