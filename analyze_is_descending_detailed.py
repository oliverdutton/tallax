#!/usr/bin/env python3
"""Detailed analysis of when is_descending is constant."""

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

def analyze_detailed():
    """Analyze when is_descending is constant with detailed rules."""

    test_cases = [
        (8, 128), (8, 256), (8, 2048), (128, 256),
    ]

    print("ANALYSIS SUMMARY")
    print("=" * 80)

    for batch_size, sort_dim in test_cases:
        num_tiles = (sort_dim * batch_size) // (NUM_SUBLANES * NUM_LANES)
        if num_tiles == 0:
            num_tiles = 1
        num_stages = int(jnp.log2(sort_dim))
        max_substage = int(jnp.log2(num_tiles * NUM_SUBLANES))

        print(f"\nShape ({batch_size}, {sort_dim}):")
        print(f"  num_tiles={num_tiles}, max_substage={max_substage}, num_stages={num_stages}")

        tile_local_offset = iota_tile(0) + (iota_tile(1) // batch_size) * num_tiles * NUM_SUBLANES

        # Determine the rule for each stage
        for stage in range(1, num_stages + 1):
            dim1_offset = 0  # ascending

            # Sample first few tiles
            is_same_all_tiles = True
            is_constant_per_tile = True

            first_tile_val = None
            for idx in range(min(num_tiles, 4)):
                tile_offset = idx * NUM_SUBLANES
                is_desc = create_bit_indicator(stage, dim1_offset + tile_offset + tile_local_offset)

                # Check if constant within tile
                if not jnp.all(is_desc == is_desc[0, 0]):
                    is_constant_per_tile = False

                # Check if same as first tile
                if first_tile_val is None:
                    first_tile_val = is_desc[0, 0]
                elif is_desc[0, 0] != first_tile_val:
                    is_same_all_tiles = False

            # Determine category
            if is_same_all_tiles and is_constant_per_tile:
                category = "SAME_ALL"
            elif is_constant_per_tile:
                category = "CONST_PER_TILE"
            else:
                category = "VARIES"

            print(f"    Stage {stage:2d}: {category:15s}", end="")

            # Determine the rule
            # The key insight: stage determines which bit we're looking at
            # - If stage < log2(NUM_SUBLANES), the bit varies within iota_tile(0) range (0-7)
            # - If stage < max_substage, the bit is set by tile_offset differences
            # - If stage >= max_substage, the bit is constant across our tile range

            # But we need to consider tile_local_offset contributions from both dims
            max_tile_local = (NUM_LANES - 1) // batch_size * num_tiles * NUM_SUBLANES + (NUM_SUBLANES - 1)

            # Check if bit at position 'stage' varies in tile_local_offset
            bit_varies_in_local = ((max_tile_local >> stage) & 1) != ((0 >> stage) & 1)

            # Check if bit at position 'stage' varies across tiles
            max_tile_offset = (num_tiles - 1) * NUM_SUBLANES
            bit_varies_in_tiles = ((max_tile_offset >> stage) & 1) != ((0 >> stage) & 1)

            if bit_varies_in_local:
                expected = "VARIES (bit varies in local)"
            elif bit_varies_in_tiles:
                expected = "CONST_PER_TILE (bit varies across tiles)"
            else:
                expected = "SAME_ALL (bit constant)"

            match = "✓" if category in expected else "✗"
            print(f" | Expected: {expected:40s} {match}")

    print("\n" + "=" * 80)
    print("OPTIMIZATION RULES:")
    print("=" * 80)
    print("""
For a given stage and configuration:
1. Compute max_tile_local_offset (max value of tile_local_offset)
2. Compute max_tile_offset = (num_tiles - 1) * NUM_SUBLANES

Then:
- If bit at 'stage' varies within [0, max_tile_local_offset]:
    → is_descending VARIES within tile → Must compute full (8, 128) array

- Else if bit at 'stage' varies within [0, max_tile_offset]:
    → is_descending is CONSTANT per tile but differs between tiles
    → Can compute single value per tile: create_bit_indicator(stage, tile_offset)

- Else:
    → is_descending is SAME for all tiles
    → Can compute single global value: create_bit_indicator(stage, 0)
    """)

if __name__ == "__main__":
    analyze_detailed()
