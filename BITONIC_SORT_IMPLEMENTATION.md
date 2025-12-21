# Bitonic Sort Implementation

This document describes the implementation of bitonic sort in tallax, which complements the existing bitonic top-k functionality.

## Overview

The bitonic sort implementation provides a full sorting capability using the same compressed transpose format and tiling strategy as bitonic top-k, optimized for TPU execution.

**Current Status:** The implementation is fully functional for arbitrary sort dimensions. It uses compressed transpose format for dimensions up to NUM_LANES (128) and extends with cross-lane permutation substages for larger dimensions. Successfully handles arrays like (8, 2048) with full tile unrolling.

## Key Components

### 1. `bitonic_sort_arrays(operands, num_keys, axis, descending)`

Main array-level function that performs bitonic sort on arrays. Similar to `bitonic_topk_arrays` but performs full sorting without reduction.

**Features:**
- Works on arbitrary input shapes (e.g., (8, 2048))
- Uses compressed transpose format for efficient TPU execution
- Supports both ascending and descending sort
- Multi-key sorting support
- Handles batch sizes > NUM_LANES by splitting and processing chunks

**Algorithm:**
1. Pad input to compatible shape
2. Convert to compressed transpose format
3. Run bitonic sort stages 1 through log2(n)
4. Convert back from compressed format
5. Unpad to original shape

### 2. `bitonic_sort(operand, num_keys, descending, interpret)`

Public API function with Pallas kernel wrapper. This is the main entry point for users.

**Example:**
```python
import jax.numpy as jnp
from tallax._src.bitonic_topk import bitonic_sort

x = jnp.array([[3, 1, 4, 2], [8, 5, 7, 6]], dtype=jnp.int32)
result = bitonic_sort(x, descending=False)
# result[0] = [[1, 2, 3, 4], [5, 6, 7, 8]]
```

### 3. `_run_bitonic_stage_on_tiles(arrs_tiles, stage, batch_size, num_keys, dim1_offset)`

Executes a complete bitonic sort stage on tiled arrays. A stage consists of multiple substages that perform comparisons at decreasing distances.

**Key differences from top-k:**
- Keeps all values (no reduction)
- Follows bitonic pattern based on stage and position
- Accepts stage parameter to determine `is_descending` dynamically

### 4. Helper Functions

- `_bitonic_reduce_inter_tile`: Performs cross-tile bitonic comparison (keeps both halves)
- `_bitonic_reduce_intra_tile`: Performs intra-tile bitonic comparison with proper `is_descending` computation

## Comparison with Bitonic Top-K

| Aspect | Bitonic Top-K | Bitonic Sort |
|--------|--------------|--------------|
| Output size | Returns top k elements | Returns all elements sorted |
| Tile reduction | Progressively reduces tiles | Maintains all tiles |
| `is_descending` | Always True (max selection) | Computed from bitonic pattern |
| Use case | Selecting top-k from large arrays | Full sorting of arrays |
| Complexity | O(n log k) comparisons | O(n log²n) comparisons |

## Implementation Details

### Stage Parameter

The `stage` parameter is crucial for determining the bitonic sort pattern. At stage `s`, we're sorting sequences of length 2^s, and the sort direction (ascending/descending) alternates based on position within the sequence.

The `dim1_offset` parameter controls whether the final result is ascending (`dim1_offset=0`) or descending (`dim1_offset=sort_dim`).

### Compressed Transpose Format

For input shape (8, 2048):
1. Split into 16 chunks of (8, 128) along axis 1
2. Concatenate along axis 0: (128, 128)
3. Transpose: (128, 128)
4. Split into 16 tiles of (8, 128)

This format optimizes for TPU's VMEM architecture and efficient permutation operations.

### Full Unroll Strategy

Similar to bitonic top-k, the implementation uses full unrolling into tiles for maximum performance:
- Stages 1-7 are fully unrolled when they fit within NUM_LANES (128)
- Each stage runs all its substages in sequence
- Compiler can fuse operations for better performance

## Testing

Comprehensive tests are provided in `tests/bitonic_sort_test.py`:
- Various shapes including (8, 2048)
- Multiple data types (int32, float32)
- Both ascending and descending sort
- Multi-key sorting
- Comparison against JAX reference implementation

## Example Usage

```python
import jax.numpy as jnp
from tallax._src.bitonic_topk import bitonic_sort

# Sort a batch of arrays
x = jax.random.randint(jax.random.PRNGKey(0), (8, 2048), 0, 1000, dtype=jnp.int32)

# Ascending sort
result_asc = bitonic_sort(x, descending=False)

# Descending sort
result_desc = bitonic_sort(x, descending=True)

# Multi-key sort (sort by first array, break ties with second)
values = jax.random.randint(jax.random.PRNGKey(0), (8, 2048), 0, 10, dtype=jnp.int32)
indices = jax.random.randint(jax.random.PRNGKey(1), (8, 2048), 0, 100, dtype=jnp.int32)
sorted_values, sorted_indices = bitonic_sort([values, indices], num_keys=2)
```

## Performance Characteristics

- Optimized for power-of-2 sized arrays
- Automatic padding for non-power-of-2 sizes
- Efficient for batch sizes up to NUM_LANES (128)
- Larger batches handled via automatic chunking
- Best performance on TPU; CPU support via interpret mode

## Implementation Highlights

### Cross-Lane Support for Large Arrays

The implementation successfully handles arbitrary sort dimensions by combining two approaches:

**For dimensions ≤ NUM_LANES (128):**
- Uses `run_compressed_transpose_format_substages_on_tiles` for all substages
- Fully optimized compressed transpose format
- Examples: (8, 16) through (128, 128) ✓

**For dimensions > NUM_LANES (e.g., 2048):**
- Substages 0 to `log2(num_tiles * NUM_SUBLANES)` use compressed format
- Remaining substages use cross-lane permutations with `iota_tile(1)`
- Examples: (8, 256), (8, 512), (8, 1024), (8, 2048) ✓

### Key Implementation Details

**Range Fix:** Critical bug fix in substage iteration:
```python
# Correct: Includes all cross-lane substages from max_substage to stage-1
for substage in range(max_substage, stage)[::-1]:
```

**Position Mapping:** Uses the same formula as sort.py:
```python
tile_local_offset = iota_tile(0) + (iota_tile(1) // batch_size) * num_tiles * NUM_SUBLANES
is_descending = create_bit_indicator(stage, dim1_offset + tile_offset + tile_local_offset)
```

**Optimization:** The `_compute_is_descending_for_tile` function optimizes `is_descending` computation following the pattern from `sort.py`:
- **SAME_2D_FOR_ALL_TILES**: When `stage < log2(NUM_SUBLANES)` (stage < 3), returns a (8,128) array based on `tile_local_offset` only, same for all tiles
- **SCALAR_PER_TILE**: When `stage < log2(num_tiles * NUM_SUBLANES)`, returns one scalar boolean per tile
- **FULL_ARRAY**: For larger stages, computes the full (8, 128) array per tile

This optimization provides significant benefits:
- (8, 2048): 54% of stages optimized (6 of 11)
- (128, 256): 87% of stages optimized (7 of 8)
- Reduces memory traffic and computation

See `OPTIMIZATION_REPORT.md` for detailed analysis.

**Descending Sort:** Controlled by `dim1_offset`:
```python
dim1_offset = int(descending) * sort_dim  # 0 for ascending, sort_dim for descending
```

### Validated Shapes

All the following shapes pass comprehensive tests with both ascending and descending sort:
- (8, 16), (8, 64), (8, 128) ✓
- (8, 256), (8, 512), (8, 1024), (8, 2048) ✓
- (16, 128), (32, 128), (64, 128), (128, 128), (128, 256) ✓
