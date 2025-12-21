#!/usr/bin/env python3
"""Correct analysis of when is_descending is constant."""

import jax.numpy as jnp
import jax

NUM_SUBLANES = 8
NUM_LANES = 128

def iota_tile(dim):
    """Create iota array with tile shape."""
    return jax.lax.broadcasted_iota(jnp.int32, (NUM_SUBLANES, NUM_LANES), dim)

def create_bit_indicator(bit_position, index):
    """Create mask indicating which elements have specific bit set."""
    bit = (index & (1 << bit_position))
    return bit > 0

def log2(x):
    return int(jnp.log2(x))

def analyze_correct():
    """Correctly analyze when is_descending is constant."""

    test_cases = [
        (8, 128), (8, 256), (8, 2048), (128, 256),
    ]

    print("DETAILED IS_DESCENDING ANALYSIS")
    print("=" * 80)

    for batch_size, sort_dim in test_cases:
        num_tiles = (sort_dim * batch_size) // (NUM_SUBLANES * NUM_LANES)
        if num_tiles == 0:
            num_tiles = 1
        num_stages = log2(sort_dim)
        max_substage = log2(num_tiles * NUM_SUBLANES)

        print(f"\nShape ({batch_size}, {sort_dim}):")
        print(f"  num_tiles={num_tiles}, max_substage={max_substage}, num_stages={num_stages}")

        tile_local_offset = iota_tile(0) + (iota_tile(1) // batch_size) * num_tiles * NUM_SUBLANES
        max_local = int(jnp.max(tile_local_offset))

        print(f"  tile_local_offset range: [0, {max_local}]")

        for stage in range(1, num_stages + 1):
            dim1_offset = 0  # ascending

            # Check ALL tiles
            is_same_all_tiles = True
            is_constant_per_tile = True

            first_tile_val = None
            for idx in range(num_tiles):  # Check ALL tiles
                tile_offset = idx * NUM_SUBLANES
                is_desc = create_bit_indicator(stage, dim1_offset + tile_offset + tile_local_offset)

                # Check if constant within tile
                if not jnp.all(is_desc == is_desc[0, 0]):
                    is_constant_per_tile = False

                # Check if same as first tile
                tile_first_val = bool(is_desc[0, 0])
                if first_tile_val is None:
                    first_tile_val = tile_first_val
                elif tile_first_val != first_tile_val:
                    is_same_all_tiles = False

            # Determine category
            if not is_constant_per_tile:
                category = "VARIES"
            elif is_same_all_tiles:
                category = "SAME_ALL"
            else:
                category = "CONST_PER_TILE"

            # Now determine the CORRECT expected value
            # Key insight: is_descending is based on bit at position 'stage'
            # in the value (tile_offset + tile_local_offset)

            # Does bit toggle in tile_local_offset range [0, max_local]?
            bit_val = 1 << stage
            toggles_in_local = max_local >= bit_val

            # Does bit toggle across tile offsets?
            max_tile_offset = (num_tiles - 1) * NUM_SUBLANES
            min_tile_offset = 0

            # Check if any tile offset has the bit set differently
            toggles_in_tiles = False
            first_bit = None
            for idx in range(num_tiles):
                tile_offset = idx * NUM_SUBLANES
                bit_set = (tile_offset & bit_val) != 0
                if first_bit is None:
                    first_bit = bit_set
                elif bit_set != first_bit:
                    toggles_in_tiles = True
                    break

            if toggles_in_local:
                expected = "VARIES"
            elif toggles_in_tiles:
                expected = "CONST_PER_TILE"
            else:
                expected = "SAME_ALL"

            match = "✓" if category == expected else "✗"
            print(f"    Stage {stage:2d}: {category:15s} | Expected: {expected:15s} {match}")

    print("\n" + "=" * 80)
    print("OPTIMIZATION RULES:")
    print("=" * 80)
    print("""
Given:
- stage: current sorting stage
- max_local: max value of tile_local_offset
- num_tiles: number of tiles
- NUM_SUBLANES = 8

Rules:
1. If max_local >= 2^stage:
     → Bit at position 'stage' varies within tile_local_offset
     → is_descending VARIES within tile
     → Must compute full (8, 128) array

2. Else, check if bit toggles across tile_offset values:
   - For each tile idx in [0, num_tiles-1]:
       - Check if (idx * NUM_SUBLANES) has bit 'stage' set
   - If bit value differs across tiles:
       → is_descending is CONSTANT per tile
       → Can compute: create_bit_indicator(stage, tile_offset + dim1_offset)

3. Else:
       → is_descending is SAME for all tiles
       → Can compute: create_bit_indicator(stage, dim1_offset)

Simplified check:
- max_local >= (1 << stage) → VARIES
- Else, check if any bit at position 'stage' differs in {idx * 8 | idx in [0, num_tiles-1]}
  - If yes → CONST_PER_TILE
  - If no → SAME_ALL
    """)

if __name__ == "__main__":
    analyze_correct()
