#!/usr/bin/env python3
"""Debug is_descending computation."""

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

# Test case: (8, 256), stage 3
batch_size = 8
sort_dim = 256
num_tiles = (sort_dim * batch_size) // (NUM_SUBLANES * NUM_LANES)
stage = 3

print(f"Testing: batch_size={batch_size}, sort_dim={sort_dim}, stage={stage}")
print(f"num_tiles={num_tiles}")

tile_local_offset = iota_tile(0) + (iota_tile(1) // batch_size) * num_tiles * NUM_SUBLANES

print(f"\ntile_local_offset shape: {tile_local_offset.shape}")
print(f"tile_local_offset[0, :16]: {tile_local_offset[0, :16]}")
print(f"tile_local_offset[:, 0]: {tile_local_offset[:, 0]}")

for idx in range(num_tiles):
    tile_offset = idx * NUM_SUBLANES
    is_desc = create_bit_indicator(stage, tile_offset + tile_local_offset)

    print(f"\nTile {idx} (offset={tile_offset}):")
    print(f"  is_desc shape: {is_desc.shape}")
    print(f"  is_desc unique values: {jnp.unique(is_desc)}")
    print(f"  is_desc[0, 0] = {is_desc[0, 0]}")
    print(f"  is_desc[7, 0] = {is_desc[7, 0]}")
    print(f"  is_desc[0, 7] = {is_desc[0, 7]}")
    print(f"  is_desc[0, 8] = {is_desc[0, 8]}")
    print(f"  is_desc[0, 127] = {is_desc[0, 127]}")
    print(f"  All equal to [0,0]? {jnp.all(is_desc == is_desc[0, 0])}")

    # Show where bit is set
    offsets = tile_offset + tile_local_offset
    print(f"  Sample offsets [0,:8]: {offsets[0, :8]}")
    print(f"  Bit {stage} set in offsets [0,:8]: {[bool(create_bit_indicator(stage, o)) for o in offsets[0, :8].tolist()]}")
