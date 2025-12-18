"""Comprehensive test to verify the new format's indexing logic."""

NUM_LANES = 128
NUM_SUBLANES = 8

def test_global_base_index_formula():
    """Test that global_base_index correctly maps to original j indices."""

    print("="*60)
    print("=== TESTING GLOBAL_BASE_INDEX FORMULA ===")
    print("="*60)
    print()

    dim0 = 8
    n = 16
    num_tiles = 16

    n_times_dim0 = num_tiles * NUM_SUBLANES
    print(f"dim0={dim0}, n={n}, num_tiles={num_tiles}, n_times_dim0={n_times_dim0}")
    print()

    # Test a few positions within a tile
    test_cases = [
        (0, 0, 0),  # tile_idx=0, row=0, col=0
        (0, 0, 64),  # tile_idx=0, row=0, col=64
        (0, 7, 127),  # tile_idx=0, row=7, col=127
        (5, 3, 50),  # tile_idx=5, row=3, col=50
        (15, 7, 127),  # tile_idx=15, row=7, col=127
    ]

    print("Testing: (tile_idx, row_in_tile, col_in_tile) -> original j")
    print()

    for tile_idx, row_in_tile, col_in_tile in test_cases:
        # New formula for global_base_index
        global_base_index = (col_in_tile // dim0) * n_times_dim0 + row_in_tile

        # Tile offset
        tile_offset = tile_idx * NUM_SUBLANES

        # Total j
        j = tile_offset + global_base_index

        # Verify by reverse mapping
        # Compressed position
        compressed_row = tile_idx * NUM_SUBLANES + row_in_tile
        compressed_col = col_in_tile

        # Transpose
        transposed_row = compressed_col
        transposed_col = compressed_row

        # Split and concat
        chunk_idx = transposed_row // dim0
        i = transposed_row % dim0
        j_verify = chunk_idx * n_times_dim0 + transposed_col

        print(f"Tile {tile_idx:2d}, pos ({row_in_tile}, {col_in_tile:3d}):")
        print(f"  Formula: tile_offset={tile_offset:3d}, global_base_index={global_base_index:4d}, j={j:4d}")
        print(f"  Verify:  compressed ({compressed_row:3d}, {compressed_col:3d}) -> " +
              f"transpose ({transposed_row:3d}, {transposed_col:3d}) -> " +
              f"original ({i}, {j_verify:4d})")

        if j == j_verify:
            print(f"  ✓ MATCH")
        else:
            print(f"  ✗ MISMATCH!")
        print()


def test_tile_offset_formula():
    """Test that tile_offset correctly computes base j for each tile."""

    print("="*60)
    print("=== TESTING TILE_OFFSET FORMULA ===")
    print("="*60)
    print()

    dim0 = 8
    n = 16
    num_tiles = 16

    print(f"dim0={dim0}, n={n}, num_tiles={num_tiles}")
    print()

    # Test each tile's offset
    print("Testing: tile_idx -> tile_offset (base j for element (0,0) in tile)")
    print()

    for tile_idx in range(num_tiles):
        # New formula
        tile_offset = tile_idx * NUM_SUBLANES

        # Verify by computing j for element (0,0) in this tile
        compressed_row = tile_idx * NUM_SUBLANES + 0
        compressed_col = 0

        # Transpose
        transposed_row = compressed_col  # 0
        transposed_col = compressed_row  # tile_idx * 8

        # Original position
        chunk_idx = transposed_row // dim0  # 0
        i = transposed_row % dim0  # 0
        j_verify = chunk_idx * (num_tiles * NUM_SUBLANES) + transposed_col

        print(f"Tile {tile_idx:2d}: offset={tile_offset:3d}, verify j={j_verify:3d}", end="")

        if tile_offset == j_verify:
            print(" ✓")
        else:
            print(" ✗ MISMATCH!")


def test_separation_formula():
    """Test that separation is correct for cross-tile comparisons."""

    print("\n" + "="*60)
    print("=== TESTING SEPARATION FORMULA ===")
    print("="*60)
    print()

    # Test for different substages
    for substage in range(3, 8):  # Substages that do cross-tile comparisons
        separation = 2**substage // NUM_SUBLANES

        print(f"Substage {substage}: separation = 2^{substage} / 8 = {separation}")
        print(f"  Tiles compared: 0 with {separation}, 1 with {1+separation}, etc.")


def test_stage_calculations():
    """Test stage calculations for bitonic topk."""

    print("\n" + "="*60)
    print("=== TESTING STAGE CALCULATIONS ===")
    print("="*60)
    print()

    dim0 = 8
    dim1 = 2048
    log_lanes = 7  # log2(128)

    print(f"dim0={dim0}, dim1={dim1}")
    print()

    # Cross-tile merge stage
    cross_tile_stage = 11  # log2(NUM_LANES * NUM_LANES // dim0)
    expected = (NUM_LANES * NUM_LANES) // dim0
    print(f"Cross-tile merge stage: log2({NUM_LANES}*{NUM_LANES}//{dim0}) = log2({expected}) = {cross_tile_stage}")
    print(f"  This checks bit {cross_tile_stage} of original j")
    print(f"  Pattern flips every {2**cross_tile_stage} elements")
    print()

    # Intra-tile merge stages
    num_merges = 11 - 7  # log2(dim1) - log2(NUM_LANES)
    num_intra_merges = 4  # min(log2(ceil(NUM_LANES/dim0)), num_merges)

    print(f"Intra-tile merges: {num_intra_merges} iterations")
    for i in range(num_intra_merges-1, -1, -1):  # Reverse order
        stage = log_lanes + i
        distance = dim0 * (2**i)

        print(f"  Iteration {num_intra_merges-1-i}: i={i}, stage={stage}, distance={distance}")
        print(f"    Checks bit {stage} of original j")
        print(f"    Pattern flips every {2**stage} elements")
        print(f"    Permutation distance in tile coords: {distance}")


def test_split_actives():
    """Test _split_actives logic for new format."""

    print("\n" + "="*60)
    print("=== TESTING SPLIT_ACTIVES ===")
    print("="*60)
    print()

    # Test with different numbers of tiles
    for num_tiles in [2, 3, 4, 5, 16, 17]:
        num_active = 2 * (num_tiles // 2)
        num_remainder = num_tiles - num_active

        print(f"num_tiles={num_tiles:2d}: active={num_active:2d}, remainder={num_remainder}")

    print()
    print("Active tiles are the first (even number) of tiles")
    print("Remainder is the last tile if total is odd, empty otherwise")


if __name__ == "__main__":
    test_global_base_index_formula()
    test_tile_offset_formula()
    test_separation_formula()
    test_stage_calculations()
    test_split_actives()
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
