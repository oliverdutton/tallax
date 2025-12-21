# Compressed Transpose Format Sorting

This document describes the new faster sorting implementation using compressed transpose format throughout.

## Overview

The new `sort_compressed` function in `tallax/_src/sort_compressed.py` provides a faster implementation of bitonic sort by minimizing transpose operations:

- **1 transpose at load**: Convert input to compressed transpose format once
- **All operations in compressed format**: All stages and substages work directly on compressed tiles
- **1 transpose at store**: Convert back to normal format once at the end

This contrasts with the original implementation which transposes to/from compressed format multiple times per stage.

## Key Benefits

1. **Fewer transposes**: Only 2 transposes (load + store) vs. many transposes in the original implementation
2. **Better compilation**: Controlled unrolling avoids excessive tile generation for large arrays
3. **Simpler dataflow**: Data stays in one format throughout the sort

## Compilation Strategy

For large sort dimensions (e.g., 2^17 = 131k elements), fully unrolling all stages would create too many tiles (2^14 = 16,384 tiles), causing very slow compilation.

The implementation uses a tiered strategy:

```
unroll_stage_limit = log2(unroll * num_sublanes)  # Default: log2(128 * 8) = 10

Stages 1 to unroll_stage_limit:
  - Fully unrolled (sequential, not in loop)
  - Allows compiler fusion
  - Fast compilation for reasonable number of tiles

Stages > unroll_stage_limit:
  - Use fori_loop to avoid unrolling too much
  - Still work entirely in compressed format
  - Trade some fusion for compilation speed
```

### Example: (8, 2^13) array

- Sort dimension: 2^13 = 8192
- Number of tiles: 2^13 / NUM_SUBLANES = 2^13 / 8 = 2^10 = 1024 tiles
- unroll_stage_limit = log2(128 * 8) = 10
- Total stages = log2(8192) = 13

Execution:
- Stages 1-10: Fully unrolled (1024 tiles, manageable)
- Stages 11-13: Use fori_loop

### Example: (8, 2^17) array

- Sort dimension: 2^17 = 131k
- Number of tiles: 2^17 / 8 = 2^14 = 16,384 tiles
- unroll_stage_limit = 10
- Total stages = 17

Execution:
- Stages 1-10: Fully unrolled (up to 1024 tiles)
- Stages 11-17: Use fori_loop (avoids creating 16k tiles at once)

## Compressed Transpose Format

In compressed transpose format, a 2D array (batch, sort_dim) is converted to a list of tiles:

```python
Input:  (batch, sort_dim)  where batch <= NUM_LANES=128
Output: list of (batch, NUM_LANES) tiles
Number of tiles = ceil(sort_dim / (NUM_LANES / batch))
```

For example, (8, 2048) becomes:
- Each tile: (8, 128)
- Number of tiles: 2048 / (128/8) = 2048 / 16 = 128 tiles

The format enables efficient TPU operations:
- **Sublane permutes (axis=0)**: Compare elements within a tile's sublanes
- **Cross-tile operations**: Compare elements across tiles
- **Lane permutes (axis=1)**: For very large substages (rarely needed)

## Usage

```python
from tallax._src.sort_compressed import sort_compressed

# Sort an (8, 8192) array
import jax.numpy as jnp
x = jnp.arange(8 * 8192, dtype=jnp.float32).reshape(8, 8192)[:, ::-1]
sorted_x, = sort_compressed(x, num_keys=1, descending=False)

# Multi-key sort
keys = [jax.random.normal(jax.random.key(i), (8, 2048)) for i in range(2)]
sorted_keys = sort_compressed(keys, num_keys=2)

# Adjust unroll factor for compilation performance
sorted_x, = sort_compressed(x, num_keys=1, unroll=256)  # More unrolling
```

## Current Limitations

1. **No stable sort**: The current implementation doesn't support `is_stable=True`
2. **No argsort**: The `return_argsort` parameter is not implemented yet
3. **Power-of-2 sort dimension**: Sort dimension must be a power of 2
4. **No top-k extraction**: The `k` parameter must equal the full sort dimension

## Future Improvements

1. **Lane permutes**: Implement cross-lane comparisons for very large substages (when substage >= log2(num_tiles * NUM_SUBLANES))
2. **Stable sort support**: Add indices tracking for stable sorting
3. **Argsort support**: Return permutation indices
4. **Dynamic dimensions**: Support non-power-of-2 dimensions through padding

## Implementation Details

### Stages and Substages

Bitonic sort operates in stages and substages:

- **Stage s**: Builds bitonic sequences of length 2^s
- **Substages of stage s**: Run substages s-1, s-2, ..., 0 to merge into sorted order

For each stage s:
1. If s <= unroll_stage_limit: Run all s substages in compressed format (fully unrolled)
2. If s > unroll_stage_limit: Use fori_loop, run substages in compressed format

### Compressed Format Operations

The implementation reuses `run_compressed_transpose_format_substages_on_tiles` from the original `sort.py`:

- Handles within-tile comparisons (sublane permutes)
- Handles cross-tile comparisons
- Computes bitonic sort direction based on stage and element position

## Testing

Run tests with:

```bash
pytest tests/sort_compressed_test.py
```

Tests cover:
- Basic sorting (ascending/descending)
- Reverse-sorted inputs (worst case)
- Multi-key sorting
- Various array sizes (128, 256, 1024, 8192)
