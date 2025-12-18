"""Test script to understand and verify the compressed transpose format change."""

NUM_LANES = 128
NUM_SUBLANES = 8

def analyze_indexing():
    """Analyze how indexing changes between old and new formats."""

    # Test with dim0=8, n=2, so shape is (8, 256)
    dim0 = 8
    n = 2
    dim1 = n * NUM_LANES  # 256

    print(f"Test array shape: ({dim0}, {dim1})")
    print(f"NUM_LANES = {NUM_LANES}, NUM_SUBLANES = {NUM_SUBLANES}")
    print()

    print("=== OLD FORMAT MAPPING ===")
    print("Steps: (b, n*128) -> split to n chunks of (b, 128) -> concat -> (n*b, 128) -> transpose -> (128, n*b)")
    print()

    test_positions = [(0, 0), (0, 128), (3, 130), (7, 255), (5, 100)]

    for i, j in test_positions:
        # Old format mapping
        chunk_idx = j // NUM_LANES
        j_in_chunk = j % NUM_LANES
        row_after_concat = chunk_idx * dim0 + i
        col_after_concat = j_in_chunk
        # After transpose
        final_row = col_after_concat
        final_col = row_after_concat

        print(f"Original ({i:2d}, {j:3d}) -> chunk {chunk_idx}, pos in chunk ({i}, {j_in_chunk})")
        print(f"  After concat: ({row_after_concat:3d}, {col_after_concat:3d})")
        print(f"  After transpose: ({final_row:3d}, {final_col:3d})")
        print()

    print("\n=== NEW FORMAT MAPPING ===")
    print("Steps: (b, n*128) -> split to (128//b) chunks along dim1 -> concat dim0 -> (128, n*b)")
    print()

    n_splits = NUM_LANES // dim0
    chunk_size = dim1 // n_splits  # This should be n * dim0

    print(f"Number of splits: {n_splits}")
    print(f"Each chunk has dim1 size: {chunk_size} (should equal n*dim0 = {n * dim0})")
    print()

    for i, j in test_positions:
        # New format mapping
        chunk_idx = j // chunk_size
        j_in_chunk = j % chunk_size
        # After concat along dim0
        final_row = chunk_idx * dim0 + i
        final_col = j_in_chunk

        print(f"Original ({i:2d}, {j:3d}) -> chunk {chunk_idx}, pos in chunk ({i}, {j_in_chunk})")
        print(f"  After concat dim0: ({final_row:3d}, {final_col:3d})")
        print()

    print("\n=== GENERAL FORMULAS ===")
    print("\nOLD FORMAT: (i, j) in (b, n*128) -> (row, col) in (128, n*b)")
    print("  row = j % 128")
    print("  col = (j // 128) * b + i")
    print()

    print("NEW FORMAT: (i, j) in (b, n*128) -> (row, col) in (128, n*b)")
    print("  n_splits = 128 // b")
    print("  chunk_size = (n*128) // n_splits = n*b")
    print("  row = (j // (n*b)) * b + i")
    print("  col = j % (n*b)")
    print()

    print("=== KEY DIFFERENCE ===")
    print("OLD: Row determined by j%128 (position within 128-chunk)")
    print("NEW: Row determined by which split chunk + original row i")
    print("OLD: Col determined by which 128-chunk and original row")
    print("NEW: Col determined by position within split chunk")


def analyze_reverse_mapping():
    """Analyze the reverse mapping (from compressed back to original)."""

    dim0 = 8
    n = 2
    dim1 = n * NUM_LANES  # 256

    print("\n" + "="*60)
    print("=== REVERSE MAPPING (from_compressed_transpose_format) ===")
    print("="*60)

    print(f"\nGoing from ({NUM_LANES}, {n * dim0}) back to ({dim0}, {dim1})")
    print()

    # Test positions in the compressed format
    test_compressed = [(0, 0), (2, 11), (67, 2), (127, 15)]

    print("OLD FORMAT REVERSE: (128, n*b) -> transpose -> (n*b, 128) -> split and concat -> (b, n*128)")
    for row, col in test_compressed:
        # Reverse transpose: (row, col) -> (col, row)
        transposed_row = col
        transposed_col = row

        # Which chunk and position within chunk
        chunk_idx = transposed_row // dim0
        i = transposed_row % dim0
        j_in_chunk = transposed_col
        j = chunk_idx * NUM_LANES + j_in_chunk

        print(f"Compressed ({row:3d}, {col:3d}) -> after transpose ({transposed_row:3d}, {transposed_col:3d}) -> original ({i:2d}, {j:3d})")

    print()
    print("NEW FORMAT REVERSE: (128, n*b) -> split to rows -> concat along dim1 -> (b, n*128)")

    n_splits = NUM_LANES // dim0
    chunk_size = n * dim0

    for row, col in test_compressed:
        # Which row chunk
        chunk_idx = row // dim0
        i = row % dim0

        # The column becomes part of the reconstructed dim1
        j = chunk_idx * chunk_size + col

        print(f"Compressed ({row:3d}, {col:3d}) -> chunk {chunk_idx}, i={i} -> original ({i:2d}, {j:3d})")


def analyze_tile_indexing():
    """Analyze how tiles are indexed in both formats."""

    dim0 = 8
    n = 2

    print("\n" + "="*60)
    print("=== TILE INDEXING ===")
    print("="*60)

    final_shape_row = NUM_LANES
    final_shape_col = n * dim0  # 16

    num_tile_rows = final_shape_row // NUM_SUBLANES  # 128 / 8 = 16
    num_tile_cols = final_shape_col // NUM_LANES  # 16 / 128 = 0.125 ???

    print(f"\nCompressed format shape: ({final_shape_row}, {final_shape_col})")
    print(f"Tile shape: ({NUM_SUBLANES}, {NUM_LANES})")
    print(f"Number of tile rows: {final_shape_row} / {NUM_SUBLANES} = {num_tile_rows}")
    print(f"Number of tile cols: {final_shape_col} / {NUM_LANES} = {num_tile_cols}")
    print()

    # Wait, this doesn't work. The final shape col is only 16, but tiles need 128 columns
    # This means we must be padding or the shape must be different

    print("NOTE: When dim1 < NUM_LANES, the final shape needs adjustment!")
    print("The code must pad dim1 to be at least NUM_LANES.")

    # Let's try with a larger example
    dim0 = 8
    n = 16  # Much larger
    dim1 = n * NUM_LANES  # 2048

    final_shape_col = n * dim0  # 128

    num_tile_rows = NUM_LANES // NUM_SUBLANES  # 128 / 8 = 16
    num_tile_cols = final_shape_col // NUM_LANES  # 128 / 128 = 1

    print(f"\nLarger example: dim0={dim0}, n={n}, dim1={dim1}")
    print(f"Compressed format shape: ({NUM_LANES}, {final_shape_col})")
    print(f"Number of tile rows: {num_tile_rows}")
    print(f"Number of tile cols: {num_tile_cols}")
    print(f"Total tiles: {num_tile_rows * num_tile_cols} = {num_tile_rows}")


if __name__ == "__main__":
    analyze_indexing()
    analyze_reverse_mapping()
    analyze_tile_indexing()
