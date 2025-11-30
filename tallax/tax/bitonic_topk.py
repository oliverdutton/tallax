"""
Bitonic Top-K for k=NUM_LANES=128 using sublane transposed format.

This implementation is optimized for TPU with k=128 and works entirely in
sublane transposed format to maximize efficiency of permutation operations.
"""

import functools
from collections.abc import Sequence

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
    canonicalize_operand,
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


def _lane_permute_merge_progressive(arrs_tiles, initial_stage, num_keys, b):
    """
    Progressive lane permute merging with decreasing distances and stages.

    Runs log2(128//b) iterations with:
    - Iteration i: distance = 64 >> i, stage = initial_stage - i
    - Each iteration: permute, max merge, run substages 6-0

    Args:
        arrs_tiles: List of tile arrays [vals_tiles, idxs_tiles]
        initial_stage: Starting stage (7 + log2(128//b))
        num_keys: Number of sort keys
        b: Block size (num_tokens)

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

    # Progressive merging: log2(128//b) iterations
    num_iterations = log2(NUM_LANES // b)

    for i in range(num_iterations):
        distance = 64 >> i  # 64, 32, 16, ..., down to b
        current_stage = initial_stage - i

        # Create permutation: XOR with distance (equivalent to roll for power-of-2 distances)
        # Element at position i gets combined with element at i XOR distance
        index = iota_tile(1)
        permutation = jnp.bitwise_xor(index, distance)

        # Permute using take_along_axis (TPU-supported for (8, 128) tiles)
        vals_permuted = jnp.take_along_axis(vals, permutation, axis=1)
        idxs_permuted = jnp.take_along_axis(idxs, permutation, axis=1)

        # Max merge: keep the larger values
        mask = vals > vals_permuted
        vals = jnp.where(mask, vals, vals_permuted)
        idxs = jnp.where(mask, idxs, idxs_permuted)

        # Run substages 6-0 for current stage
        vals_tiles = split_array_to_tiles(vals)
        idxs_tiles = split_array_to_tiles(idxs)

        arrs_tiles = _compute_subtile_substages_inner(
            [vals_tiles, idxs_tiles],
            num_substages=7,  # substages 0-6
            stage=current_stage,
            b=b,
            use_lane_permute=False,
            num_keys=num_keys,
            dim1_offset=0,
        )

        vals_tiles, idxs_tiles = arrs_tiles
        vals = join_tiles_to_array(shape, vals_tiles)
        idxs = join_tiles_to_array(shape, idxs_tiles)

    # Convert back to tiles
    vals_tiles = split_array_to_tiles(vals)
    idxs_tiles = split_array_to_tiles(idxs)

    return [vals_tiles, idxs_tiles]


def bitonic_topk_kernel(
    in_refs,
    out_refs,
    *,
    num_keys: int,
    descending: bool,
):
    """
    Pallas kernel for bitonic top-k with k=128 in sublane format.

    Algorithm:
    1. Convert to sublane transposed format: (num_tokens, vocab) -> (128, num_tokens*chunks)
    2. Run stages 1-6 to sort up to 64 within each lane
    3. Run substages 6-0 for stage 7+log2(128//b) to create bitonic sequences
    4. While tiles >= 32: merge tile pairs using max, then run substages
    5. When tiles <= 16: progressive lane permute merging:
       - log2(128//b) iterations with decreasing distance (64, 32, ..., b)
       - Each iteration: roll permute, max merge, run substages for decreasing stage
    6. Transpose back from sublane format and extract top-128 per token
    """
    num_tokens = in_refs[0].shape[0]
    vocab_size = in_refs[0].shape[1]
    b = num_tokens

    if b > NUM_LANES:
        raise ValueError(f"num_tokens must be <= NUM_LANES, got {num_tokens}")

    # Calculate dim1_offset for descending sort
    dim1_offset = int(descending) * vocab_size

    # Convert to sublane format: (num_tokens, vocab_size) -> (128, num_tokens * num_chunks)
    # where num_chunks = vocab_size // 128
    num_chunks = vocab_size // NUM_LANES

    # Process all input operands
    operands_sublane = []

    for in_ref in in_refs:
        # Stack chunks: for each 128-wide chunk, stack all tokens
        chunks = []
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * NUM_LANES
            chunk = in_ref[:, pl.dslice(chunk_start, NUM_LANES)]  # (num_tokens, 128)
            chunks.append(chunk)

        # Concatenate chunks and transpose to sublane format
        # (num_tokens, vocab_size) -> (vocab_size, num_tokens) -> (128, num_chunks * num_tokens)
        full = jnp.concatenate(chunks, axis=1).T  # (vocab_size, num_tokens)

        # Reshape to sublane format: (128, num_chunks * num_tokens)
        reshaped = []
        for i in range(num_chunks):
            reshaped.append(full[i * NUM_LANES:(i + 1) * NUM_LANES, :])

        sublane = jnp.concatenate(reshaped, axis=1)  # (128, num_chunks * num_tokens)
        operands_sublane.append(sublane)

    # Pad to power of 2
    target_dim1 = 2 ** log2(operands_sublane[0].shape[1])
    if target_dim1 < NUM_LANES:
        target_dim1 = NUM_LANES

    if operands_sublane[0].shape[1] < target_dim1:
        pad_size = target_dim1 - operands_sublane[0].shape[1]
        operands_sublane = tuple(
            jnp.pad(
                op,
                ((0, 0), (0, pad_size)),
                constant_values=jnp.finfo(op.dtype).min if jnp.issubdtype(op.dtype, jnp.floating) else -1
            )
            for op in operands_sublane
        )

    # Convert to tiles for processing
    arrs_tiles = tuple(split_array_to_tiles(op) for op in operands_sublane)

    # Stage 1: Bitonic sort stages 1-6 (sort up to 64 within each lane)
    arrs_tiles = _compute_subtile_substages_inner(
        list(arrs_tiles),
        num_substages=6,
        stage=None,  # None means run stages 1-6 as fused block
        b=b,
        use_lane_permute=False,
        num_keys=num_keys,
        dim1_offset=dim1_offset,
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
        num_keys=num_keys,
        dim1_offset=dim1_offset,
    )

    # Stage 3: Iteratively merge tiles
    # While we have >= 32 tiles (16 pairs), merge using max
    num_tiles = len(arrs_tiles[0])

    while num_tiles >= 32:  # Can merge 16 pairs of tiles
        # Merge consecutive pairs using max
        arrs_tiles = _merge_tiles_max(
            arrs_tiles,
            stage=merge_stage,
            dim1_offset=dim1_offset,
            num_keys=num_keys,
            b=b
        )

        # Run substages again after merge
        arrs_tiles = _compute_subtile_substages_inner(
            arrs_tiles,
            num_substages=7,
            stage=merge_stage,
            b=b,
            use_lane_permute=False,
            num_keys=num_keys,
            dim1_offset=dim1_offset,
        )

        num_tiles = len(arrs_tiles[0])

    # Stage 4: When <= 16 tiles remain, use progressive lane permutes
    # Only do this if b < NUM_LANES (otherwise no merging needed)
    if b < NUM_LANES and num_tiles <= NUM_LANES // NUM_SUBLANES:
        arrs_tiles = _lane_permute_merge_progressive(
            arrs_tiles,
            initial_stage=merge_stage,
            num_keys=num_keys,
            b=b
        )

    # Extract results and convert back from sublane format
    final_shape = (NUM_LANES, len(arrs_tiles[0]) * NUM_SUBLANES)

    # In sublane sorted format:
    # - Row i contains the i-th ranked value from each token
    # - First b columns contain values for tokens 0 to b-1
    # - Next b columns contain values for tokens 0 to b-1 (next chunk), etc.
    # After all merging, the first b columns should have the top-128 per token
    # Transpose to get (num_tokens, 128)
    for tiles, out_ref in zip(arrs_tiles, out_refs):
        final = join_tiles_to_array(final_shape, tiles)  # (128, ...)
        out_ref[...] = final[:, :num_tokens].T  # (num_tokens, 128)


@functools.partial(
    jit,
    static_argnames=("k", "num_keys", "descending", "interpret"),
)
def bitonic_topk(
    operand: jax.Array | Sequence[jax.Array],
    k: int = NUM_LANES,
    num_keys: int = 1,
    descending: bool = True,
    interpret: bool = False,
) -> tuple[jax.Array, ...]:
    """
    Compute top-k using bitonic sort in sublane transposed format.

    Optimized for k=NUM_LANES=128 only. Works entirely in sublane transposed
    format for maximum TPU efficiency. Supports multiple operands like sort().

    Args:
        operand: Input array(s) of shape [num_tokens, vocab_size].
                Can be a single array or sequence of arrays.
                vocab_size must be a multiple of NUM_LANES.
        k: Number of top elements (must be NUM_LANES=128).
        num_keys: Number of arrays to use as sort keys.
        descending: If True, sort in descending order (default for top-k).
        interpret: If True, run in CPU interpret mode.

    Returns:
        Tuple of arrays (same length as input operands):
            - Each array has shape [num_tokens, k]

    Raises:
        ValueError: If k != NUM_LANES, vocab_size not multiple of NUM_LANES,
                   or num_tokens > NUM_LANES
    """
    if k != NUM_LANES:
        raise ValueError(
            f"bitonic_topk only supports k=NUM_LANES={NUM_LANES}, got k={k}"
        )

    operands, shape = canonicalize_operand(operand)
    num_tokens, vocab_size = shape

    if vocab_size % NUM_LANES != 0:
        raise ValueError(
            f"vocab_size must be multiple of NUM_LANES={NUM_LANES}, got {vocab_size}"
        )

    if num_tokens > NUM_LANES:
        raise ValueError(
            f"num_tokens must be <= NUM_LANES={NUM_LANES}, got {num_tokens}"
        )

    # Pad operands to proper dimensions
    operands = tuple(
        pad(x, block_shape=(NUM_SUBLANES, 'power_of_2_lanes'), prepend=(False, descending))
        for x in operands
    )

    # Update shape after padding
    padded_vocab_size = operands[0].shape[1]

    # Define output shapes
    output_shapes = tuple(
        jax.ShapeDtypeStruct((num_tokens, NUM_LANES), op.dtype)
        for op in operands
    )

    outputs = pl.pallas_call(
        functools.partial(
            bitonic_topk_kernel,
            num_keys=num_keys,
            descending=descending,
        ),
        in_specs=(tuple(
            pl.BlockSpec((num_tokens, padded_vocab_size), lambda: (0, 0))
            for _ in operands
        ),),
        out_shape=(output_shapes,),
        out_specs=(tuple(
            pl.BlockSpec()
            for _ in output_shapes
        ),),
        grid=(),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=int(0.9 * 2**27)
        ),
        interpret=interpret,
    )(operands)

    # Unpad if needed (extract k elements from the correct side)
    if not descending:
        outputs = tuple(x[:, :k] for x in outputs)
    else:
        # For descending, top-k is at the beginning after sorting
        outputs = tuple(x[:, :k] for x in outputs)

    return outputs
