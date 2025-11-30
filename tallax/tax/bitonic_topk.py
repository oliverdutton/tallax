"""
Bitonic Top-K for k=NUM_LANES=128 using sublane transposed format.

This implementation is optimized for TPU with k=128 and works entirely in
sublane transposed format to maximize efficiency of permutation operations.
"""

import functools
import jax
import jax.numpy as jnp
from jax import lax, jit
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax.utils import (
    NUM_LANES,
    NUM_SUBLANES,
    log2,
    iota_tile,
    create_bit_indicator,
    split_array_to_tiles,
    join_tiles_to_array,
    pad,
)
from tallax.tax.sort import (
    _compute_subtile_substages_inner,
    _compute_substage_by_crosstile_comparison,
)


def _merge_tiles_max(arrs_tiles, stage, dim1_offset, num_keys, b):
    """
    Merge consecutive pairs of tiles using max operation.

    Takes pairs of tiles and keeps the maximum values, reducing tile count by half.

    Args:
        arrs_tiles: List of tile arrays [vals_tiles, idxs_tiles]
        stage: Current bitonic stage
        dim1_offset: Offset for bitonic direction computation
        num_keys: Number of sort keys
        b: Block size (number of tokens)

    Returns:
        List of merged tile arrays
    """
    vals_tiles, idxs_tiles = arrs_tiles
    num_tiles = len(vals_tiles)

    if num_tiles % 2 != 0:
        raise ValueError(f"Cannot merge odd number of tiles: {num_tiles}")

    merged_vals = []
    merged_idxs = []

    # Process pairs of consecutive tiles
    for i in range(0, num_tiles, 2):
        left_val = vals_tiles[i]
        right_val = vals_tiles[i + 1]
        left_idx = idxs_tiles[i]
        right_idx = idxs_tiles[i + 1]

        # Element-wise max
        mask = left_val > right_val
        merged_val = jnp.where(mask, left_val, right_val)
        merged_idx = jnp.where(mask, left_idx, right_idx)

        merged_vals.append(merged_val)
        merged_idxs.append(merged_idx)

    return [merged_vals, merged_idxs]


def _lane_permute_merge(arrs_tiles, stage, dim1_offset, num_keys, b):
    """
    Merge tiles using lane permutations via take_along_axis.

    Used when fewer tiles remain and we need cross-tile communication.

    Args:
        arrs_tiles: List of tile arrays [vals_tiles, idxs_tiles]
        stage: Current bitonic stage
        dim1_offset: Offset for bitonic direction
        num_keys: Number of sort keys
        b: Block size

    Returns:
        List of merged tile arrays
    """
    vals_tiles, idxs_tiles = arrs_tiles
    num_tiles = len(vals_tiles)

    # Reconstruct arrays from tiles
    tile_rows = NUM_LANES // NUM_SUBLANES
    tile_cols = num_tiles // tile_rows
    shape = (NUM_LANES, tile_cols * NUM_LANES)

    vals = join_tiles_to_array(shape, vals_tiles)
    idxs = join_tiles_to_array(shape, idxs_tiles)

    # Perform bitonic merge using lane permutations (supported by TPU via take_along_axis)
    # This is an (8, 128) permute pattern
    for substage in range(log2(NUM_LANES))[::-1]:
        index = iota_tile(1)
        permutation = jnp.bitwise_xor(index, 1 << substage)

        # Take along axis is supported for (8, 128) tiles
        vals_permuted = jnp.take_along_axis(vals, permutation, axis=1)
        idxs_permuted = jnp.take_along_axis(idxs, permutation, axis=1)

        # Compare and swap
        is_right_half = create_bit_indicator(substage, index)
        mask = vals > vals_permuted
        mask = jnp.bitwise_xor(mask.astype(jnp.int32), is_right_half.astype(jnp.int32)).astype(bool)

        vals = jnp.where(mask, vals, vals_permuted)
        idxs = jnp.where(mask, idxs, idxs_permuted)

    # Convert back to tiles
    vals_tiles = split_array_to_tiles(vals)
    idxs_tiles = split_array_to_tiles(idxs)

    return [vals_tiles, idxs_tiles]


def bitonic_topk_kernel(
    logits_ref,
    topk_vals_ref,
    topk_idxs_ref,
):
    """
    Pallas kernel for bitonic top-k with k=128 in sublane format.

    Algorithm:
    1. Convert to sublane transposed format
    2. Sort up to 128 within each lane (stages 1-6)
    3. Run substages 6-0 for stage 7+log2(128//b)
    4. Merge tiles using max operations (for 16+ tiles)
    5. Merge remaining tiles using lane permutes
    6. Extract top-128 and convert back
    """
    num_tokens = logits_ref.shape[0]
    vocab_size = logits_ref.shape[1]
    b = num_tokens

    if b > NUM_LANES:
        raise ValueError(f"num_tokens must be <= NUM_LANES, got {num_tokens}")

    # Convert to sublane format: (num_tokens, vocab_size) -> (128, num_tokens * num_chunks)
    # where num_chunks = vocab_size // 128
    num_chunks = vocab_size // NUM_LANES

    # Stack chunks: for each 128-wide chunk, stack all tokens
    vals_chunks = []
    idxs_chunks = []

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * NUM_LANES
        chunk_vals = logits_ref[:, pl.dslice(chunk_start, NUM_LANES)]  # (num_tokens, 128)

        # Create indices
        base_idx = jnp.full((num_tokens, NUM_LANES), chunk_start, dtype=jnp.int32)
        chunk_idxs = base_idx + jax.lax.broadcasted_iota(jnp.int32, (num_tokens, NUM_LANES), 1)

        vals_chunks.append(chunk_vals)
        idxs_chunks.append(chunk_idxs)

    # Concatenate chunks and transpose to sublane format
    # (num_tokens, vocab_size) -> (vocab_size, num_tokens) -> (128, num_chunks * num_tokens)
    vals_full = jnp.concatenate(vals_chunks, axis=1).T  # (vocab_size, num_tokens)
    idxs_full = jnp.concatenate(idxs_chunks, axis=1).T

    # Reshape to sublane format: (128, num_chunks * num_tokens)
    vals_reshaped = []
    idxs_reshaped = []
    for i in range(num_chunks):
        vals_reshaped.append(vals_full[i * NUM_LANES:(i + 1) * NUM_LANES, :])
        idxs_reshaped.append(idxs_full[i * NUM_LANES:(i + 1) * NUM_LANES, :])

    vals_sublane = jnp.concatenate(vals_reshaped, axis=1)  # (128, num_chunks * num_tokens)
    idxs_sublane = jnp.concatenate(idxs_reshaped, axis=1)

    # Pad to power of 2
    target_dim1 = 2 ** log2(vals_sublane.shape[1])
    if target_dim1 < NUM_LANES:
        target_dim1 = NUM_LANES

    if vals_sublane.shape[1] < target_dim1:
        pad_size = target_dim1 - vals_sublane.shape[1]
        vals_sublane = jnp.pad(
            vals_sublane,
            ((0, 0), (0, pad_size)),
            constant_values=jnp.finfo(jnp.float32).min
        )
        idxs_sublane = jnp.pad(
            idxs_sublane,
            ((0, 0), (0, pad_size)),
            constant_values=-1
        )

    # Convert to tiles for processing
    vals_tiles = split_array_to_tiles(vals_sublane)
    idxs_tiles = split_array_to_tiles(idxs_sublane)

    arrs_tiles = [vals_tiles, idxs_tiles]

    # Stage 1: Bitonic sort stages 1-6 (sort up to 64 within each lane)
    arrs_tiles = _compute_subtile_substages_inner(
        arrs_tiles,
        num_substages=6,
        stage=None,  # None means run stages 1-6 as fused block
        b=b,
        use_lane_permute=False,
        num_keys=1,
        dim1_offset=0,
    )

    # Stage 2: Run substages for higher stage
    # stage = 7 + log2(128 // b)
    merge_stage = 7 + log2(NUM_LANES // b)

    # Run substages 6 down to 0 for this stage
    arrs_tiles = _compute_subtile_substages_inner(
        arrs_tiles,
        num_substages=7,  # Run substages 0-6 (that's 7 substages)
        stage=merge_stage,
        b=b,
        use_lane_permute=False,
        num_keys=1,
        dim1_offset=0,
    )

    # Stage 3: Iteratively merge tiles
    # While we have >= 32 tiles (16 pairs), merge using max
    num_tiles = len(arrs_tiles[0])

    while num_tiles >= 32:  # Can merge 16 pairs of tiles
        # Merge consecutive pairs using max
        arrs_tiles = _merge_tiles_max(
            arrs_tiles,
            stage=merge_stage,
            dim1_offset=0,
            num_keys=1,
            b=b
        )

        # Run substages again after merge
        arrs_tiles = _compute_subtile_substages_inner(
            arrs_tiles,
            num_substages=7,
            stage=merge_stage,
            b=b,
            use_lane_permute=False,
            num_keys=1,
            dim1_offset=0,
        )

        num_tiles = len(arrs_tiles[0])

    # Stage 4: When < 16 tiles remain, use lane permutes
    if num_tiles > NUM_LANES // NUM_SUBLANES:  # More than 16 tiles
        arrs_tiles = _lane_permute_merge(
            arrs_tiles,
            stage=merge_stage,
            dim1_offset=0,
            num_keys=1,
            b=b
        )

    # Extract results and convert back from sublane format
    vals_tiles, idxs_tiles = arrs_tiles
    final_shape = (NUM_LANES, len(vals_tiles) * NUM_SUBLANES)

    vals_final = join_tiles_to_array(final_shape, vals_tiles)  # (128, ...)
    idxs_final = join_tiles_to_array(final_shape, idxs_tiles)

    # The top-k values for each token are now distributed across the sublane format
    # We need to extract them properly

    # In sublane sorted format, row i contains the i-th value from each token's top-k
    # So we just need the first 128 columns, transposed
    topk_vals_ref[...] = vals_final[:, :num_tokens].T  # (num_tokens, 128)
    topk_idxs_ref[...] = idxs_final[:, :num_tokens].T


@functools.partial(
    jit,
    static_argnames=("k", "interpret"),
)
def bitonic_topk(
    logits: jax.Array,
    k: int = NUM_LANES,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute top-k using bitonic sort in sublane transposed format.

    Optimized for k=NUM_LANES=128 only. Works entirely in sublane transposed
    format for maximum TPU efficiency.

    Args:
        logits: Input logits of shape [num_tokens, vocab_size].
                vocab_size must be a multiple of NUM_LANES.
        k: Number of top elements (must be NUM_LANES=128).
        interpret: If True, run in CPU interpret mode.

    Returns:
        Tuple of (topk_vals, topk_idxs):
            - topk_vals: Top-k values of shape [num_tokens, k]
            - topk_idxs: Top-k indices of shape [num_tokens, k]

    Raises:
        ValueError: If k != NUM_LANES or vocab_size not multiple of NUM_LANES
    """
    if k != NUM_LANES:
        raise ValueError(
            f"bitonic_topk only supports k=NUM_LANES={NUM_LANES}, got k={k}"
        )

    num_tokens, vocab_size = logits.shape

    if vocab_size % NUM_LANES != 0:
        raise ValueError(
            f"vocab_size must be multiple of NUM_LANES={NUM_LANES}, got {vocab_size}"
        )

    if num_tokens > NUM_LANES:
        raise ValueError(
            f"num_tokens must be <= NUM_LANES={NUM_LANES}, got {num_tokens}"
        )

    # Define output shapes
    output_shapes = (
        jax.ShapeDtypeStruct((num_tokens, NUM_LANES), logits.dtype),
        jax.ShapeDtypeStruct((num_tokens, NUM_LANES), jnp.int32),
    )

    topk_vals, topk_idxs = pl.pallas_call(
        bitonic_topk_kernel,
        in_specs=(
            pl.BlockSpec((num_tokens, vocab_size), lambda: (0, 0)),
        ),
        out_shape=output_shapes,
        out_specs=(
            pl.BlockSpec(),
            pl.BlockSpec(),
        ),
        grid=(),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=int(0.9 * 2**27)
        ),
        interpret=interpret,
    )(logits)

    return topk_vals, topk_idxs
