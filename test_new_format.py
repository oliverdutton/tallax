"""Test the new compressed transpose format with transpose."""

NUM_LANES = 128
NUM_SUBLANES = 8

def test_format_mapping():
    """Test element mapping in new format."""

    dim0 = 8
    n = 2
    dim1 = n * NUM_LANES  # 256

    print(f"Original shape: ({dim0}, {dim1})")
    print()

    print("NEW FORMAT STEPS:")
    print(f"1. Split ({dim0}, {dim1}) into {NUM_LANES//dim0} chunks along dim1")
    print(f"   Each chunk: ({dim0}, {n * dim0})")
    print(f"2. Concat along dim0: ({NUM_LANES}, {n * dim0})")
    print(f"3. Transpose: ({n * dim0}, {NUM_LANES})")
    print(f"4. Split to tiles of ({NUM_SUBLANES}, {NUM_LANES})")
    print()

    # Final shape is (n*dim0, 128) = (16, 128)
    final_shape = (n * dim0, NUM_LANES)
    tile_rows = final_shape[0] // NUM_SUBLANES  # 16/8 = 2
    tile_cols = final_shape[1] // NUM_LANES  # 128/128 = 1

    print(f"Final shape: {final_shape}")
    print(f"Tile layout: {tile_rows} rows x {tile_cols} cols")
    print(f"Total tiles: {tile_rows * tile_cols}")
    print()

    # Test specific elements
    test_positions = [(0, 0), (0, 128), (3, 130), (7, 255), (5, 100)]

    print("Element mapping:")
    for i, j in test_positions:
        # Step 1: Split - which chunk?
        n_splits = NUM_LANES // dim0  # 16
        chunk_size = dim1 // n_splits  # 256 / 16 = 16
        chunk_idx = j // chunk_size
        j_in_chunk = j % chunk_size

        # Step 2: After concat on dim0
        row_after_concat = chunk_idx * dim0 + i
        col_after_concat = j_in_chunk

        # Step 3: After transpose
        final_row = col_after_concat
        final_col = row_after_concat

        # Which tile?
        tile_row = final_row // NUM_SUBLANES
        tile_col = final_col // NUM_LANES
        row_in_tile = final_row % NUM_SUBLANES
        col_in_tile = final_col % NUM_LANES

        print(f"({i:2d}, {j:3d}) -> chunk {chunk_idx:2d} -> concat ({row_after_concat:3d}, {col_after_concat:3d}) " +
              f"-> transpose ({final_row:3d}, {final_col:3d}) -> tile({tile_row}, {tile_col}) pos ({row_in_tile}, {col_in_tile:3d})")


def test_reverse_mapping():
    """Test reverse mapping."""

    dim0 = 8
    n = 2
    dim1 = n * NUM_LANES

    final_shape = (n * dim0, NUM_LANES)

    print("\nREVERSE MAPPING:")
    print(f"From {final_shape} back to ({dim0}, {dim1})")
    print()

    # Test positions in compressed format
    test_compressed = [(0, 0), (2, 67), (15, 127), (11, 13)]

    for row, col in test_compressed:
        # Step 1: Transpose (row, col) -> (col, row)
        transposed_row = col
        transposed_col = row

        # Step 2: Split and concat
        # transposed shape is (128, 16)
        chunk_idx = transposed_row // dim0
        i = transposed_row % dim0
        j = chunk_idx * (n * dim0) + transposed_col

        print(f"Compressed ({row:2d}, {col:3d}) -> transpose ({transposed_row:3d}, {transposed_col:3d}) " +
              f"-> split chunk {chunk_idx} -> original ({i}, {j:3d})")


def analyze_global_base_index_new():
    """Analyze global_base_index for new format."""

    dim0 = 8
    n = 16

    print("\n" + "="*60)
    print("=== GLOBAL_BASE_INDEX (NEW FORMAT) ===")
    print("="*60)

    final_shape = (n * dim0, NUM_LANES)  # (128, 128)
    tile_rows = final_shape[0] // NUM_SUBLANES  # 16
    tile_cols = final_shape[1] // NUM_LANES  # 1

    print(f"\nFinal compressed shape: {final_shape}")
    print(f"Tile layout: {tile_rows} rows x {tile_cols} cols")
    print()

    # For a tile at position (tile_row_idx, tile_col_idx)
    # Element at (row_in_tile, col_in_tile) has compressed position:
    # compressed_row = tile_row_idx * 8 + row_in_tile
    # compressed_col = tile_col_idx * 128 + col_in_tile

    # Need to map back to original j column index

    tile_row_idx = 5
    tile_col_idx = 0

    print(f"Example tile: ({tile_row_idx}, {tile_col_idx})")
    print()

    # Element at (0, 0) in this tile
    compressed_row = tile_row_idx * NUM_SUBLANES + 0  # 40
    compressed_col = tile_col_idx * NUM_LANES + 0  # 0

    # Reverse: transpose
    transposed_row = compressed_col  # 0
    transposed_col = compressed_row  # 40

    # Split and concat
    chunk_idx = transposed_row // dim0  # 0
    i = transposed_row % dim0  # 0
    j = chunk_idx * (n * dim0) + transposed_col  # 0 * 128 + 40 = 40

    print(f"Element (0, 0) in tile -> compressed ({compressed_row}, {compressed_col})")
    print(f"  -> transpose ({transposed_row}, {transposed_col})")
    print(f"  -> original ({i}, {j})")
    print()

    print("For general element (row_in_tile, col_in_tile):")
    print("  compressed_row = tile_row_idx * 8 + row_in_tile")
    print("  compressed_col = tile_col_idx * 128 + col_in_tile")
    print("  transposed_row = compressed_col = tile_col_idx * 128 + col_in_tile = tile_col_idx * 128 + iota_tile(1)")
    print("  transposed_col = compressed_row = tile_row_idx * 8 + row_in_tile = tile_row_idx * 8 + iota_tile(0)")
    print("  chunk_idx = transposed_row // dim0 = (tile_col_idx * 128 + iota_tile(1)) // dim0")
    print("  j = chunk_idx * (n * dim0) + transposed_col")
    print("    = ((tile_col_idx * 128 + iota_tile(1)) // dim0) * (n * dim0) + tile_row_idx * 8 + iota_tile(0)")
    print()

    print("If dim0 divides 128 evenly (dim0 in {1,2,4,8,16,32,64,128}):")
    print("  When col_in_tile < 128 and tile_col_idx * 128 % dim0 == 0:")
    print("  chunk_idx_base = tile_col_idx * 128 // dim0")
    print("  chunk_idx = chunk_idx_base + iota_tile(1) // dim0")
    print("  j = (chunk_idx_base + iota_tile(1) // dim0) * (n * dim0) + tile_row_idx * 8 + iota_tile(0)")
    print("    = chunk_idx_base * (n * dim0) + (iota_tile(1) // dim0) * (n * dim0) + tile_row_idx * 8 + iota_tile(0)")
    print()

    print("For dim0 = 8:")
    print("  chunk_idx_base = tile_col_idx * 16")
    print("  j = tile_col_idx * 16 * (n * 8) + (iota_tile(1) // 8) * (n * 8) + tile_row_idx * 8 + iota_tile(0)")
    print("    = tile_col_idx * (n * 128) + (iota_tile(1) // 8) * (n * 8) + tile_row_idx * 8 + iota_tile(0)")
    print()

    print("GLOBAL_BASE_INDEX formula (dim0 = 8):")
    print("  global_base_index = tile_col_idx * (n * 128) + (iota_tile(1) // 8) * (n * 8) + tile_row_idx * 8 + iota_tile(0)")


def analyze_tile_offset_new():
    """Analyze tile_offset for new format."""

    dim0 = 8
    n = 16

    print("\n" + "="*60)
    print("=== TILE_OFFSET (NEW FORMAT) ===")
    print("="*60)

    final_shape = (n * dim0, NUM_LANES)  # (128, 128)
    tile_rows = final_shape[0] // NUM_SUBLANES  # 16
    tile_cols = final_shape[1] // NUM_LANES  # 1

    print(f"\nTile grid: {tile_rows} rows x {tile_cols} cols")
    print()

    # Test tile idx 5
    idx = 5
    tile_row = idx // tile_cols  # 5
    tile_col = idx % tile_cols  # 0

    print(f"Tile idx {idx} -> ({tile_row}, {tile_col})")

    # Element (0, 0) in this tile has compressed position
    compressed_row = tile_row * NUM_SUBLANES  # 40
    compressed_col = tile_col * NUM_LANES  # 0

    # Reverse to original
    transposed_row = compressed_col  # 0
    transposed_col = compressed_row  # 40

    chunk_idx = transposed_row // dim0  # 0
    i = transposed_row % dim0  # 0
    j = chunk_idx * (n * dim0) + transposed_col  # 40

    print(f"Element (0, 0) -> original ({i}, {j})")
    print(f"tile_offset should be j = {j}")
    print()

    print("TILE_OFFSET formula:")
    print("  tile_offset = chunk_idx * (n * dim0) + transposed_col")
    print("  where:")
    print("    transposed_row = tile_col * NUM_LANES")
    print("    transposed_col = tile_row * NUM_SUBLANES")
    print("    chunk_idx = transposed_row // dim0 = (tile_col * NUM_LANES) // dim0")
    print()
    print("  For dim0 dividing NUM_LANES:")
    print("    tile_offset = (tile_col * NUM_LANES // dim0) * (n * dim0) + tile_row * NUM_SUBLANES")
    print("              = tile_col * (n * NUM_LANES) + tile_row * NUM_SUBLANES")
    print()
    print("  Using flat index:")
    print("    tile_offset = (idx % tile_cols) * (n * NUM_LANES) + (idx // tile_cols) * NUM_SUBLANES")


if __name__ == "__main__":
    test_format_mapping()
    test_reverse_mapping()
    analyze_global_base_index_new()
    analyze_tile_offset_new()
