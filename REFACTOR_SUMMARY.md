# Compressed Transpose Format Refactoring Summary

## Overview

Refactored the compressed transpose format to change from concat-transpose to split-concat-transpose approach. This alters how elements are laid out in the compressed format and requires updating all indexing logic.

## Format Change

### Old Format
```
(b, n*128) → split to chunks of 128 → concat dim0 → (n*b, 128) → transpose → (128, n*b) → split to tiles
```

### New Format
```
(b, n*128) → split dim1 into (128//b) chunks → concat dim0 → (128, n*b) → transpose → (n*b, 128) → split to tiles
```

### Key Difference

- **Old**: Make 128 in dim1 BEFORE transpose → final shape (128, n*b)
- **New**: Make 128 in dim0 BEFORE transpose → final shape (n*b, 128)

This swaps the final dimensions, changing the tile grid layout from (16 rows × tile_cols cols) to (num_tiles rows × 1 col).

## File Changes

### 1. `tallax/_src/utils.py`

#### `to_compressed_transpose_format` (lines 281-309)
```python
# Old approach:
# - Split along dim1 into chunks of 128
# - Concat to (n*b, 128)
# - Transpose to (128, n*b)

# New approach:
# - Split along dim1 into (128//b) chunks
# - Concat to (128, n*b)
# - Transpose to (n*b, 128)
```

**Element mapping:**
- Old: `(i, j)` → `(j % 128, (j // 128) * b + i)`
- New: `(i, j)` → `(j % (n*b), (j // (n*b)) * b + i)`

#### `from_compressed_transpose_format` (lines 312-336)
```python
# New reverse process:
# - Transpose (n*b, 128) → (128, n*b)
# - Split into (128//b) chunks along dim0
# - Concat along dim1 → (b, n*128)
```

### 2. `tallax/_src/sort.py`

#### `_run_compressed_transpose_format_substage_on_tiles` (lines 121-168)

**Tile grid layout** (lines 122-126):
```python
# Old: tile_rows = 16, tile_cols = num_tiles // 16
# New: tile_rows = num_tiles, tile_cols = 1
tile_rows = num_tiles
tile_cols = 1
```

**Global base index** (lines 128-133):
```python
# Maps tile position (row, col) to original array column index j
# Old: iota_tile(0) + ((iota_tile(1) // dim0) * NUM_LANES)
# New: (iota_tile(1) // dim0) * n_times_dim0 + iota_tile(0)
n_times_dim0 = num_tiles * NUM_SUBLANES
global_base_index = (iota_tile(1) // dim0) * n_times_dim0 + iota_tile(0)
```

**Tile offset** (lines 135-139):
```python
# Base j index for element (0,0) in each tile
# Old: ((idx // tile_cols) * NUM_SUBLANES + (idx % tile_cols) * (NUM_LANES * (NUM_LANES // dim0)))
# New: idx * NUM_SUBLANES
tile_offset = idx * NUM_SUBLANES
```

**Separation** (lines 165-168):
```python
# Distance between tiles being compared
# Old: (2**substage // NUM_SUBLANES) * tile_cols
# New: 2**substage // NUM_SUBLANES
separation = 2**substage // NUM_SUBLANES
```

### 3. `tallax/_src/bitonic_topk.py`

#### `_split_rows` (lines 100-103)
```python
# Old: Split tiles into 16 rows of tile grid
# New: Each tile is its own "row" (1-column grid)
def _split_rows(tiles):
  return [[tile] for tile in tiles]
```

#### `_split_actives` (lines 106-113)
```python
# Old: From each grid row, take even number of tiles
# New: Take even number of tiles from flat list
def _split_actives(tiles):
  num_tiles = len(tiles)
  num_active_tiles = 2 * (num_tiles // 2)
  active = tiles[:num_active_tiles]
  remainder = tiles[num_active_tiles:]
  return [active, remainder]
```

#### `_merge_remainder` (lines 115-118)
```python
# Old: Flatten and interleave row-wise
# New: Simple concatenation
def _merge_remainder(merged, remainder):
  return merged + remainder
```

#### Cross-tile merge remainder check (line 231)
```python
# Old: has_remainder = ((len(arrs_tiles[0][::16])%2) != 0)
# New: has_remainder = ((len(arrs_tiles[0]) % 2) != 0)
```

## Indexing Formulas

### Global Base Index
Maps element position within tile to original array column index:

```python
n_times_dim0 = num_tiles * NUM_SUBLANES
global_base_index = (iota_tile(1) // dim0) * n_times_dim0 + iota_tile(0)
```

For element at (row_in_tile, col_in_tile) in tile:
```
j_component = (col_in_tile // dim0) * (n*dim0) + row_in_tile
```

### Tile Offset
Base j index for element (0,0) in each tile:

```python
tile_offset = tile_idx * NUM_SUBLANES
```

### Complete Mapping
Compressed position (tile_idx, row_in_tile, col_in_tile) → original (i, j):

1. Compressed coords: `(tile_idx * 8 + row_in_tile, col_in_tile)`
2. After transpose: `(col_in_tile, tile_idx * 8 + row_in_tile)`
3. Split and concat:
   ```python
   chunk_idx = col_in_tile // dim0
   i = col_in_tile % dim0
   j = chunk_idx * (n*dim0) + (tile_idx * 8 + row_in_tile)
   ```

## Stage Calculations (Unchanged)

The bitonic sort stage calculations remain the same because they operate on original array indices:

- **Cross-tile merge**: `stage = log2(NUM_LANES * NUM_LANES // dim0)` = 11 for dim0=8
- **Intra-tile merge**: `stage = log_lanes + i` where i ∈ [0, num_intra_merges-1]

These check bit positions in the original j index to determine ascending/descending order.

## Verification

All indexing formulas verified through test scripts:
- `test_format_change.py` - Analyzes old vs new format mapping
- `test_new_format.py` - Tests new format with transpose
- `test_indexing_logic.py` - Analyzes indexing requirements
- `test_verify_logic.py` - Comprehensive verification (all tests pass ✓)

## Testing

Run existing tests with:
```bash
python -m pytest tests/bitonic_topk_test.py -v
python -m pytest tests/sort_test.py -v
```

Tests verify:
- Bitonic topk produces correct top-k elements
- Sort produces correctly sorted arrays
- Various array shapes and dtypes work correctly
