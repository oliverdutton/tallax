#!/usr/bin/env python3
"""Analyze when is_descending is constant."""

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

def analyze_is_descending_patterns():
    """Analyze when is_descending is constant."""

    test_cases = [
        # (batch_size, sort_dim, description)
        (8, 128, "Small batch, fits in NUM_LANES"),
        (8, 256, "Small batch, first cross-lane"),
        (8, 2048, "Small batch, many cross-lane"),
        (128, 256, "Large batch, first cross-lane"),
    ]

    for batch_size, sort_dim, desc in test_cases:
        print(f"\n{'='*70}")
        print(f"{desc}: batch_size={batch_size}, sort_dim={sort_dim}")
        print('='*70)

        # Compute parameters
        num_tiles = (sort_dim * batch_size) // (NUM_SUBLANES * NUM_LANES)
        if num_tiles == 0:
            num_tiles = 1
        num_stages = int(jnp.log2(sort_dim))
        max_substage = int(jnp.log2(num_tiles * NUM_SUBLANES))

        print(f"num_tiles={num_tiles}, num_stages={num_stages}, max_substage={max_substage}")

        # Compute tile_local_offset
        tile_local_offset = iota_tile(0) + (iota_tile(1) // batch_size) * num_tiles * NUM_SUBLANES

        # Test each stage
        for stage in range(1, num_stages + 1):
            dim1_offset = 0  # ascending sort

            # Check for different tiles (idx 0, 1, 2...)
            same_across_all_tiles = True
            constant_per_tile = True

            tile_results = []
            for idx in range(min(num_tiles, 4)):  # Check first few tiles
                tile_offset = idx * NUM_SUBLANES
                is_desc = create_bit_indicator(stage, dim1_offset + tile_offset + tile_local_offset)
                tile_results.append(is_desc)

                # Check if constant within this tile
                if not jnp.all(is_desc == is_desc[0, 0]):
                    constant_per_tile = False

            # Check if same across tiles
            if len(tile_results) > 1:
                for i in range(1, len(tile_results)):
                    if not jnp.array_equal(tile_results[0], tile_results[i]):
                        same_across_all_tiles = False
                        break

            # Determine pattern
            if same_across_all_tiles:
                pattern = "SAME FOR ALL TILES"
            elif constant_per_tile:
                pattern = "CONSTANT PER TILE"
            else:
                pattern = "VARIES WITHIN TILE"

            # Determine rule
            if stage < int(jnp.log2(NUM_SUBLANES)):
                expected = "VARIES WITHIN TILE (stage < log2(NUM_SUBLANES))"
            elif stage < max_substage:
                expected = "CONSTANT PER TILE (stage < max_substage)"
            elif stage < int(jnp.log2(sort_dim)):
                expected = "SAME FOR ALL TILES (stage < log2(sort_dim))"
            else:
                expected = "FINAL STAGE"

            match = "✓" if pattern in expected or expected in pattern else "?"
            print(f"  Stage {stage}: {pattern:30s} | Expected: {expected:40s} {match}")

if __name__ == "__main__":
    analyze_is_descending_patterns()
