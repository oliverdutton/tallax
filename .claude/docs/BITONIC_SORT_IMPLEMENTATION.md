# Bitonic Sort Implementation

## Overview

Full bitonic sort implementation using compressed transpose format and tiling strategy for TPU execution. Handles arbitrary sort dimensions including large arrays like (8, 2048) and (16, 16384).

## Key Functions

### `bitonic_sort_arrays(operands, num_keys, axis, descending)`
Main array-level function performing full bitonic sort.

**Algorithm:**
1. Pad input to compatible shape
2. Convert to compressed transpose format
3. Run bitonic sort stages 1 through log2(n)
4. Convert back and unpad

### `_run_bitonic_stage_on_tiles(arrs_tiles, stage, batch_size, num_keys, dim1_offset, sort_dim)`
Executes a complete bitonic sort stage on tiled arrays.

**For stage ≤ max_substage:**
- Uses `run_compressed_transpose_format_substages_on_tiles` (compressed format)

**For stage > max_substage:**
- Cross-lane permutations for substages max_substage+1 to stage
- Uses `iota_tile(1)` permutations: `jnp.bitwise_xor(iota_tile(1), lane_separation)`
- Followed by remaining compressed format substages

## Algorithm Details

### Bitonic Pattern
At stage `s`, sorting sequences of length 2^s with ascending/descending direction based on position.

**Position Mapping:**
```python
tile_local_offset = iota_tile(0) + (iota_tile(1) // batch_size) * num_tiles * NUM_SUBLANES
is_descending = create_bit_indicator(stage, dim1_offset + tile_offset + tile_local_offset)
```

### is_descending Optimization

Stratified optimization achieving 100% coverage:

1. **stage < log2(NUM_SUBLANES)**: SAME_2D_FOR_ALL
   - `create_bit_indicator(stage, tile_local_offset + dim1_offset)`

2. **stage < log2(num_tiles * NUM_SUBLANES)**: SCALAR_PER_TILE
   - `create_bit_indicator(stage, tile_offset + dim1_offset)`

3. **stage < log2(sort_dim)**: GLOBAL_2D
   - `create_bit_indicator(stage, dim1_offset + tile_local_offset)`

4. **stage >= log2(sort_dim)**: GLOBAL_SCALAR
   - `create_bit_indicator(stage, dim1_offset)`

See `OPTIMIZATION_REPORT.md` for detailed analysis.

### Descending Sort
```python
dim1_offset = int(descending) * sort_dim  # 0 for ascending, sort_dim for descending
```

## Compressed Transpose Format

For input shape (8, 2048):
1. Split into 16 chunks of (8, 128) along axis 1
2. Concatenate along axis 0: (128, 128)
3. Transpose: (128, 128)
4. Split into 16 tiles of (8, 128)

Optimizes for TPU VMEM architecture and efficient permutation operations.

## Cross-Lane Support

**Dimensions ≤ NUM_LANES (128):**
- All substages in compressed format

**Dimensions > NUM_LANES:**
- High substages use cross-lane permutations
- Low substages use compressed format
- Critical: Substage range `range(max_substage, stage)[::-1]` includes all necessary substages

## Validated Configurations

Tested on: (8, 16) through (16, 16384)
- Both ascending and descending
- Multiple data types (int32, float32)
- Multi-key sorting
