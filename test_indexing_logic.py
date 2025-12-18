"""Test to understand how indexing changes in the new compressed transpose format."""

NUM_LANES = 128
NUM_SUBLANES = 8

def analyze_global_base_index():
    """Analyze how global_base_index should be calculated in the new format."""

    print("="*60)
    print("=== GLOBAL_BASE_INDEX CALCULATION ===")
    print("="*60)

    # Consider a position (row, col) within a tile
    # Tiles are (8, 128) each
    # We need to find the global index in the ORIGINAL array (b, n*128)

    dim0 = 8
    n = 16
    dim1 = n * NUM_LANES  # 2048

    print(f"\nOriginal array: ({dim0}, {dim1})")
    print(f"Compressed format: (128, {n * dim0})")
    print()

    # In the compressed format after tiling:
    # - tile_rows = 128 // 8 = 16
    # - tile_cols = (n*dim0) // 128 = (16*8) // 128 = 1

    tile_rows = NUM_LANES // NUM_SUBLANES
    tile_cols = (n * dim0) // NUM_LANES

    print(f"Tile layout: {tile_rows} rows x {tile_cols} cols")
    print()

    print("OLD FORMAT global_base_index calculation:")
    print("  global_base_index = iota_tile(0) + ((iota_tile(1) // dim0) * NUM_LANES)")
    print("  For tile at position (tile_row_idx, tile_col_idx):")
    print("    - Row within tile: iota_tile(0) = 0..7")
    print("    - Col within tile: iota_tile(1) = 0..127")
    print("    - global col = tile_col_idx * 128 + iota_tile(1)")
    print("    - global row in transposed space = tile_row_idx * 8 + iota_tile(0)")
    print("    - Need to map back to original space")
    print()

    # Let's trace through a specific example
    tile_row_idx = 3
    tile_col_idx = 0
    row_in_tile = 5
    col_in_tile = 67

    print(f"Example: tile({tile_row_idx}, {tile_col_idx}), position ({row_in_tile}, {col_in_tile})")
    print()

    # Old format: the compressed space is (128, n*b)
    # Position in compressed space
    compressed_row = tile_row_idx * NUM_SUBLANES + row_in_tile
    compressed_col = tile_col_idx * NUM_LANES + col_in_tile

    print(f"OLD FORMAT:")
    print(f"  Compressed position: ({compressed_row}, {compressed_col})")

    # Map back to original using old reverse formula
    # Old reverse: (row, col) in (128, n*b) -> transpose -> (col, row) -> split and concat
    transposed_row = compressed_col
    transposed_col = compressed_row
    chunk_idx_old = transposed_row // dim0
    i_old = transposed_row % dim0
    j_in_chunk_old = transposed_col
    j_old = chunk_idx_old * NUM_LANES + j_in_chunk_old

    print(f"  After transpose: ({transposed_row}, {transposed_col})")
    print(f"  Original position: ({i_old}, {j_old})")
    print()

    print(f"NEW FORMAT:")
    print(f"  Compressed position: ({compressed_row}, {compressed_col})")

    # Map back using new reverse formula
    # New reverse: (row, col) in (128, n*b) -> split rows -> concat dim1
    chunk_idx_new = compressed_row // dim0
    i_new = compressed_row % dim0
    j_new = chunk_idx_new * (n * dim0) + compressed_col

    print(f"  Original position: ({i_new}, {j_new})")
    print()

    # The global_base_index should represent the column index in the ORIGINAL array
    # for elements in the tile

    print("NEW FORMAT global_base_index formula:")
    print("  For element at (row_in_tile, col_in_tile) in tile (tile_row_idx, tile_col_idx):")
    print("    compressed_row = tile_row_idx * 8 + row_in_tile")
    print("    compressed_col = tile_col_idx * 128 + col_in_tile")
    print("    chunk_idx = compressed_row // dim0")
    print("    i = compressed_row % dim0")
    print("    j = chunk_idx * (n * dim0) + compressed_col")
    print()
    print("  Simplifying for the tile:")
    print("    chunk_idx = (tile_row_idx * 8 + iota_tile(0)) // dim0")
    print("    j_offset_from_chunk = chunk_idx * (n * dim0)")
    print("    global_base_index = j_offset_from_chunk + tile_col_idx * 128 + iota_tile(1)")
    print()
    print("  If dim0 = 8:")
    print("    chunk_idx = tile_row_idx (since row_in_tile < 8)")
    print("    global_base_index = tile_row_idx * (n * 8) + tile_col_idx * 128 + iota_tile(1)")


def analyze_tile_offset():
    """Analyze how tile_offset should be calculated for is_descending."""

    print("\n" + "="*60)
    print("=== TILE_OFFSET CALCULATION ===")
    print("="*60)

    dim0 = 8
    n = 16

    tile_rows = NUM_LANES // NUM_SUBLANES  # 16
    tile_cols = (n * dim0) // NUM_LANES  # 1

    print(f"\nTile grid: {tile_rows} rows x {tile_cols} cols")
    print()

    print("OLD FORMAT tile_offset:")
    print("  idx is the flat tile index (row-major)")
    print("  tile_offset = ((idx // tile_cols) * NUM_SUBLANES +")
    print("                 (idx % tile_cols) * (NUM_LANES * (NUM_LANES // dim0)))")
    print()

    print("What is tile_offset representing?")
    print("  It's the base column index (in original array) for the first element of the tile")
    print()

    # Let's trace through a specific tile
    idx = 5  # tile index 5
    tile_row = idx // tile_cols  # 5
    tile_col = idx % tile_cols   # 0

    print(f"Example: tile idx={idx} -> ({tile_row}, {tile_col})")
    print()

    # OLD format calculation
    tile_offset_old = (tile_row * NUM_SUBLANES + tile_col * (NUM_LANES * (NUM_LANES // dim0)))
    print(f"OLD tile_offset = {tile_row} * 8 + {tile_col} * (128 * 16) = {tile_offset_old}")

    # For OLD format, element at (0, 0) in this tile has compressed position:
    compressed_row_old = tile_row * NUM_SUBLANES  # 40
    compressed_col_old = tile_col * NUM_LANES  # 0
    # Original j position using old formula:
    # j = (compressed_row_old) % 128, then after transpose magic...
    # Actually the tile_offset is the j column index for the element at tile position (0, 0)

    # Let me recalculate properly using the mapping
    # Compressed (40, 0) in old format maps to original position:
    transposed_row_old = compressed_col_old  # 0
    transposed_col_old = compressed_row_old  # 40
    chunk_idx_old = transposed_row_old // dim0  # 0
    i_old = transposed_row_old % dim0  # 0
    j_old = chunk_idx_old * NUM_LANES + transposed_col_old  # 40

    print(f"  Element at tile position (0, 0) has compressed pos ({compressed_row_old}, {compressed_col_old})")
    print(f"  Which maps to original ({i_old}, {j_old})")
    print(f"  So tile_offset should be {j_old}")
    print()

    # NEW format calculation
    # For NEW format, element at (0, 0) in tile (5, 0):
    compressed_row_new = tile_row * NUM_SUBLANES  # 40
    compressed_col_new = tile_col * NUM_LANES  # 0

    chunk_idx_new = compressed_row_new // dim0  # 5
    i_new = compressed_row_new % dim0  # 0
    j_new = chunk_idx_new * (n * dim0) + compressed_col_new  # 5 * 128 = 640

    print(f"NEW format:")
    print(f"  Element at tile position (0, 0) has compressed pos ({compressed_row_new}, {compressed_col_new})")
    print(f"  Which maps to original ({i_new}, {j_new})")
    print(f"  So NEW tile_offset = {j_new}")
    print()

    print("NEW tile_offset formula:")
    print("  tile_offset = (tile_row * NUM_SUBLANES // dim0) * (n * dim0) + tile_col * NUM_LANES")
    print("  If dim0 = NUM_SUBLANES:")
    print("    tile_offset = tile_row * (n * dim0) + tile_col * NUM_LANES")
    print("    tile_offset = (idx // tile_cols) * (n * dim0) + (idx % tile_cols) * NUM_LANES")


def analyze_bitonic_topk_indexing():
    """Analyze the ::num_lanes/num_sublanes indexing in bitonic topk."""

    print("\n" + "="*60)
    print("=== BITONIC TOPK INDEXING ===")
    print("="*60)

    dim0 = 8
    n = 16

    tile_rows = NUM_LANES // NUM_SUBLANES  # 16
    tile_cols = (n * dim0) // NUM_LANES  # 1
    total_tiles = tile_rows * tile_cols  # 16

    print(f"\nTotal tiles: {total_tiles}")
    print()

    print("OLD FORMAT:")
    print("  has_remainder = ((len(arrs_tiles[0][::16])%2) != 0)")
    print("  This is checking tiles[::16] - every 16th tile")
    print(f"  For {total_tiles} tiles, tiles[::16] = tiles at indices [0]")
    print("  Length = 1, so has_remainder = False")
    print()

    print("What does [::16] represent in the old format?")
    print("  stride of 16 = NUM_LANES // NUM_SUBLANES")
    print("  This selects one tile per 'row' of tiles")
    print("  In old format with tile_cols=1, this selects one tile every 16 tiles")
    print()

    print("NEW FORMAT:")
    print("  The tiles are laid out differently!")
    print("  We need to reconsider what this indexing means")
    print()

    # In the new format, tiles represent different parts of the data
    # Let's think about what we're trying to check

    print("Key insight: we're checking if we have an odd number of 'tile rows'")
    print("  This is for the cross-tile merging logic")
    print()
    print("In NEW format:")
    print(f"  tile_rows = {tile_rows}, tile_cols = {tile_cols}")
    print("  We should check tiles[::tile_cols] to get one tile per row")
    print(f"  has_remainder = ((len(arrs_tiles[0][::{tile_cols}]) % 2) != 0)")
    print()

    # Actually, looking at the code more carefully, the ::16 is used
    # in the context of cross-tile merging where we merge pairs
    # We need to understand the merging strategy

    print("Actually, need to look at _merge_max_crosstile logic:")
    print("  It merges tiles pairwise (idx and idx+1)")
    print("  The ::16 indexing selects representative tiles to check if we have an odd number")
    print()
    print("The stride should be: NUM_LANES // NUM_SUBLANES = tile_rows")
    print("  This is CORRECT in both formats!")
    print("  But wait - let me reconsider...")
    print()

    # Actually the issue is more subtle
    # In the new format, the tile layout changes
    # tiles are in row-major order: tile_idx = row * tile_cols + col

    print("Tiles are indexed in row-major order:")
    print("  tile_idx = tile_row * tile_cols + tile_col")
    print()
    print(f"For tile_cols = {tile_cols}:")
    print("  tiles[::1] = all tiles (since tile_cols=1)")
    print("  To select one per row, we need stride = tile_cols")
    print()

    # Hmm, but in the old code it uses ::16
    # Let me re-examine the old format layout

    print("\nRe-examining OLD format tile layout:")
    dim0_old = 8
    n_old = 16

    # Old compressed shape after transpose: (128, n*b) = (128, 128)
    # Split into tiles of (8, 128)
    # tile_rows = 128/8 = 16
    # tile_cols = 128/128 = 1

    print(f"  Compressed shape: (128, {n_old * dim0_old})")
    print(f"  tile_rows = {NUM_LANES // NUM_SUBLANES}, tile_cols = {(n_old * dim0_old) // NUM_LANES}")
    print("  tiles[::16] selects every 16th tile = tiles [0]")
    print()

    # I think the key issue is that NUM_LANES // NUM_SUBLANES might not be the right stride
    # Let me look at what the code is actually checking


if __name__ == "__main__":
    analyze_global_base_index()
    analyze_tile_offset()
    analyze_bitonic_topk_indexing()
