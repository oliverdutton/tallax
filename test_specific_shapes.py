"""Test specific shapes: (8, 4096), (16, 2048), (128, 256)"""

NUM_LANES = 128
NUM_SUBLANES = 8

def analyze_shape(dim0, dim1):
    """Analyze a specific shape through the format conversion."""

    print(f"\n{'='*70}")
    print(f"SHAPE ({dim0}, {dim1})")
    print('='*70)

    n = dim1 // NUM_LANES
    total_elements = dim0 * dim1

    print(f"n = {n}, total elements = {total_elements}")
    print()

    # Format conversion steps
    n_splits = NUM_LANES // dim0
    chunk_dim1 = dim1 // n_splits

    print(f"Format conversion:")
    print(f"  1. Split ({dim0}, {dim1}) into {n_splits} chunks along dim1")
    print(f"     Each chunk: ({dim0}, {chunk_dim1})")
    print(f"  2. Concat dim0: ({NUM_LANES}, {chunk_dim1})")
    print(f"  3. Transpose: ({chunk_dim1}, {NUM_LANES})")

    final_shape = (chunk_dim1, NUM_LANES)
    num_tiles = (chunk_dim1 * NUM_LANES) // (NUM_SUBLANES * NUM_LANES)
    tile_rows = chunk_dim1 // NUM_SUBLANES
    tile_cols = NUM_LANES // NUM_LANES

    print(f"  4. Split to tiles ({NUM_SUBLANES}, {NUM_LANES})")
    print(f"     Tile grid: {tile_rows} rows × {tile_cols} col")
    print(f"     Total tiles: {num_tiles}")
    print()

    # Verify total elements
    tiles_elements = num_tiles * NUM_SUBLANES * NUM_LANES
    print(f"Verification: {num_tiles} tiles × {NUM_SUBLANES} × {NUM_LANES} = {tiles_elements}", end="")
    if tiles_elements == total_elements:
        print(" ✓")
    else:
        print(f" ✗ (expected {total_elements})")
    print()

    # Indexing parameters
    n_times_dim0 = num_tiles * NUM_SUBLANES
    print(f"Indexing parameters:")
    print(f"  n_times_dim0 = {num_tiles} × {NUM_SUBLANES} = {n_times_dim0}")
    print()

    # Test a specific element
    tile_idx = min(10, num_tiles - 1)
    row_in_tile = 3
    col_in_tile = 67

    print(f"Test element at tile {tile_idx}, position ({row_in_tile}, {col_in_tile}):")

    # Using formula
    tile_offset = tile_idx * NUM_SUBLANES
    global_base_index = (col_in_tile // dim0) * n_times_dim0 + row_in_tile
    j_formula = tile_offset + global_base_index

    print(f"  Formula:")
    print(f"    tile_offset = {tile_idx} × {NUM_SUBLANES} = {tile_offset}")
    print(f"    global_base_index = ({col_in_tile} // {dim0}) × {n_times_dim0} + {row_in_tile} = {global_base_index}")
    print(f"    j = {tile_offset} + {global_base_index} = {j_formula}")

    # Verify by reverse mapping
    compressed_row = tile_idx * NUM_SUBLANES + row_in_tile
    compressed_col = col_in_tile

    transposed_row = compressed_col
    transposed_col = compressed_row

    chunk_idx = transposed_row // dim0
    i = transposed_row % dim0
    j_verify = chunk_idx * n_times_dim0 + transposed_col

    print(f"  Verify:")
    print(f"    Compressed ({compressed_row}, {compressed_col}) → transpose ({transposed_row}, {transposed_col})")
    print(f"    chunk_idx = {transposed_row} // {dim0} = {chunk_idx}")
    print(f"    i = {transposed_row} % {dim0} = {i}")
    print(f"    j = {chunk_idx} × {n_times_dim0} + {transposed_col} = {j_verify}")

    if j_formula == j_verify:
        print(f"  ✓ MATCH (j = {j_formula})")
    else:
        print(f"  ✗ MISMATCH (formula: {j_formula}, verify: {j_verify})")
    print()

    # Bitonic topk parameters
    log_lanes = 7  # log2(128)
    if dim1 >= NUM_LANES:
        num_merges = len(bin(dim1)) - 3 - log_lanes  # log2(dim1) - 7
    else:
        num_merges = 0

    # Calculate log2 of NUM_LANES // dim0
    ratio = NUM_LANES // dim0
    if ratio > 0:
        log_ratio = len(bin(ratio)) - 3
    else:
        log_ratio = 0

    num_intra_merges = min(log_ratio, num_merges) if num_merges > 0 else 0
    num_cross_tile_merges = num_merges - num_intra_merges

    print(f"Bitonic topk parameters:")
    print(f"  num_merges = log2({dim1}) - 7 = {num_merges}")
    print(f"  num_intra_merges = min(log2({NUM_LANES}//{dim0}), {num_merges}) = {num_intra_merges}")
    print(f"  Cross-tile merges: {num_cross_tile_merges}")

    if num_cross_tile_merges > 0:
        cross_tile_stage = len(bin(NUM_LANES * NUM_LANES // dim0)) - 3
        print(f"  Cross-tile stage = log2({NUM_LANES}×{NUM_LANES}//{dim0}) = {cross_tile_stage}")
        print(f"    Checks bit {cross_tile_stage} of j (pattern flips every {2**cross_tile_stage})")

    if num_intra_merges > 0:
        print(f"  Intra-tile stages:")
        for i in range(num_intra_merges - 1, -1, -1):
            stage = log_lanes + i
            distance = dim0 * (2**i)
            print(f"    i={i}: stage={stage}, distance={distance}")


def main():
    """Test the three specific shapes."""

    print("="*70)
    print("TESTING SPECIFIC SHAPES FOR NEW COMPRESSED TRANSPOSE FORMAT")
    print("="*70)

    # Test each shape
    analyze_shape(8, 4096)
    analyze_shape(16, 2048)
    analyze_shape(128, 256)

    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print('='*70)
    print()

    shapes = [(8, 4096), (16, 2048), (128, 256)]

    print(f"{'Shape':<15} {'Final':<15} {'Tiles':<8} {'n_times_dim0':<12} {'Cross-tile':<12} {'Intra'}")
    print("-" * 70)

    for dim0, dim1 in shapes:
        n_splits = NUM_LANES // dim0
        chunk_dim1 = dim1 // n_splits
        final_shape = f"({chunk_dim1},{NUM_LANES})"
        num_tiles = chunk_dim1 // NUM_SUBLANES
        n_times_dim0 = num_tiles * NUM_SUBLANES

        log_lanes = 7
        num_merges = len(bin(dim1)) - 3 - log_lanes if dim1 >= NUM_LANES else 0
        ratio = NUM_LANES // dim0
        log_ratio = len(bin(ratio)) - 3 if ratio > 0 else 0
        num_intra = min(log_ratio, num_merges) if num_merges > 0 else 0
        num_cross = num_merges - num_intra

        print(f"({dim0:3},{dim1:4}){'':<4} {final_shape:<15} {num_tiles:<8} {n_times_dim0:<12} {num_cross:<12} {num_intra}")

    print()
    print("Key observation: All shapes have the same n_times_dim0=256")
    print("because they all have the same total element count (32768)")
    print("and are split into 32 tiles.")


if __name__ == "__main__":
    main()
