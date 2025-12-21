#!/usr/bin/env python3
"""Comprehensive analysis of is_descending optimization opportunities."""

import numpy as np
from collections import defaultdict

NUM_SUBLANES = 8
NUM_LANES = 128

def iota_tile(dim):
    """Create iota array with tile shape."""
    if dim == 0:
        return np.arange(NUM_SUBLANES, dtype=np.int32)[:, None]
    else:
        return np.arange(NUM_LANES, dtype=np.int32)[None, :]

def create_bit_indicator(bit_position, index):
    """Create mask indicating which elements have specific bit set."""
    bit = (index & (1 << bit_position))
    return bit > 0

def log2(x):
    return int(np.log2(x))

def analyze_is_descending_patterns(batch_size, sort_dim, descending=False):
    """Analyze is_descending patterns for a given configuration."""

    num_tiles = (sort_dim * batch_size) // (NUM_SUBLANES * NUM_LANES)
    if num_tiles == 0:
        num_tiles = 1
    num_stages = log2(sort_dim)
    max_substage = log2(num_tiles * NUM_SUBLANES)

    dim1_offset = int(descending) * sort_dim
    tile_local_offset = iota_tile(0) + (iota_tile(1) // batch_size) * num_tiles * NUM_SUBLANES

    print(f"\n{'='*80}")
    print(f"Configuration: ({batch_size}, {sort_dim}), descending={descending}")
    print(f"{'='*80}")
    print(f"num_tiles={num_tiles}, max_substage={max_substage}, num_stages={num_stages}")
    print(f"dim1_offset={dim1_offset}")

    for stage in range(1, num_stages + 1):
        # Compute is_descending for all tiles
        all_tiles = []
        for idx in range(num_tiles):
            tile_offset = idx * NUM_SUBLANES
            is_desc = create_bit_indicator(stage, dim1_offset + tile_offset + tile_local_offset)
            all_tiles.append(is_desc)

        # Analysis 1: Check if each tile is constant (scalar)
        tiles_are_scalar = [np.all(tile == tile.flat[0]) for tile in all_tiles]
        all_scalar = all(tiles_are_scalar)

        # Analysis 2: If scalar, check uniqueness across tiles
        unique_values = None
        if all_scalar:
            scalar_values = [tile.flat[0] for tile in all_tiles]
            unique_values = len(set(scalar_values))

        # Analysis 3: Check uniqueness of full tiles (as patterns)
        unique_tiles = {}
        for idx, tile in enumerate(all_tiles):
            tile_hash = tile.tobytes()
            if tile_hash not in unique_tiles:
                unique_tiles[tile_hash] = []
            unique_tiles[tile_hash].append(idx)
        num_unique_tiles = len(unique_tiles)

        # Analysis 4: Check if all tiles are identical
        all_same = num_unique_tiles == 1

        # Analysis 5: Check if pattern repeats (useful for modulo indexing)
        pattern_period = None
        if num_unique_tiles < num_tiles:
            # Try to find repeating pattern
            for period in range(1, num_tiles):
                if num_tiles % period == 0:
                    # Check if tiles repeat with this period
                    repeats = True
                    for i in range(num_tiles):
                        if all_tiles[i].tobytes() != all_tiles[i % period].tobytes():
                            repeats = False
                            break
                    if repeats:
                        pattern_period = period
                        break

        # Determine optimization strategy
        strategy = "FULL_ARRAY"
        details = ""

        if all_same:
            if tiles_are_scalar[0]:
                strategy = "GLOBAL_SCALAR"
                details = f"value={all_tiles[0].flat[0]}"
            else:
                strategy = "GLOBAL_2D"
                details = f"shape={all_tiles[0].shape}"
        elif all_scalar:
            strategy = "SCALAR_PER_TILE"
            details = f"{unique_values} unique values across {num_tiles} tiles"
            if pattern_period:
                details += f", period={pattern_period}"
        elif num_unique_tiles < num_tiles:
            strategy = "SHARED_PATTERNS"
            details = f"{num_unique_tiles} unique patterns across {num_tiles} tiles"
            if pattern_period:
                details += f", period={pattern_period}"

        # Compare with threshold-based rules
        threshold_strategy = "FULL_ARRAY"
        if stage < log2(NUM_SUBLANES):
            threshold_strategy = "SAME_2D_FOR_ALL"
        elif stage < log2(num_tiles * NUM_SUBLANES):
            threshold_strategy = "SCALAR_PER_TILE"

        match = "✓" if strategy == threshold_strategy or (
            strategy == "GLOBAL_SCALAR" and threshold_strategy == "SCALAR_PER_TILE"
        ) or (
            strategy == "GLOBAL_2D" and threshold_strategy == "SAME_2D_FOR_ALL"
        ) else "✗"

        print(f"  Stage {stage:2d}: {strategy:20s} | {details:50s} | Threshold: {threshold_strategy:20s} {match}")

        # Show tile pattern if useful
        if pattern_period and pattern_period < min(16, num_tiles):
            print(f"           Pattern repeats every {pattern_period} tiles:")
            for i in range(pattern_period):
                if tiles_are_scalar[i]:
                    print(f"             Tile {i}: scalar={all_tiles[i].flat[0]}")
                else:
                    print(f"             Tile {i}: 2D array with {np.sum(all_tiles[i])} True values")

def main():
    """Run comprehensive analysis."""

    test_cases = [
        (8, 128),
        (8, 256),
        (8, 512),
        (8, 1024),
        (8, 2048),
        (16, 4096),
        (16, 16384),
        (128, 256),
        (128, 512),
        (256, 256),
    ]

    for batch_size, sort_dim in test_cases:
        analyze_is_descending_patterns(batch_size, sort_dim, descending=False)

    print("\n" + "="*80)
    print("SUMMARY OF OPTIMIZATION OPPORTUNITIES")
    print("="*80)
    print("""
Based on the analysis above, we can identify several optimization strategies:

1. GLOBAL_SCALAR: Single bool for all tiles
   - Most efficient, single value reused everywhere

2. GLOBAL_2D: Single (8,128) array for all tiles
   - Compute once, reuse for all tiles
   - Occurs when stage < log2(NUM_SUBLANES)

3. SCALAR_PER_TILE: One bool per tile
   - Store in 1D array of length num_tiles
   - Use tile index to look up value
   - Occurs when stage < log2(num_tiles * NUM_SUBLANES)

4. SHARED_PATTERNS with period: Pre-compute unique patterns
   - If num_unique < num_tiles and pattern repeats
   - Store unique patterns in array
   - Use (tile_idx % period) to index

5. FULL_ARRAY: Full (8,128) per tile
   - Compute for each tile independently
   - Necessary when values vary within and across tiles

Threshold-based rules from sort.py capture cases 1-3 effectively.
Cases 4-5 could benefit from additional pattern detection.
    """)

if __name__ == "__main__":
    main()
