"""
Bitonic Top-K for k=NUM_LANES=128 using sublane transposed format.

This implementation is optimized for TPU with k=128 and works entirely in
sublane transposed format to maximize efficiency of permutation operations.
"""

import functools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax.utils import (
    NUM_LANES,
    NUM_SUBLANES,
    log2,
    iota_tile,
    split_array_to_tiles,
    join_tiles_to_array,
    pad,
    canonicalize_operand,
    transpose_list_of_lists,
    convert_to_sublane_sort_format,
    convert_from_sublane_sort_format,
)
from tallax.tax.sort import (
    _compute_subtile_substages_inner,
    _compare,
    _compute_start_index,
)


def _merge_tiles_max(arrs_tiles, num_keys, separation=16, is_descending=True):
    """
    Merge tile pairs with given separation using max (or min) operation.

    Args:
        arrs_tiles: Tuple of lists of tile arrays [tiles_op0, tiles_op1, ...]
        num_keys: Number of sort keys
        separation: Distance between tiles to compare (default 16 for blocks of 32)
        is_descending: If True, keep max. If False, keep min.

    Returns:
        Tuple of lists of merged tile arrays
    """
    num_tiles = len(arrs_tiles[0])

    if num_tiles % 2 != 0:
        raise ValueError(f"Cannot merge odd number of tiles: {num_tiles}")

    merged_tiles = [[] for _ in arrs_tiles]

    # Process pairs using _compute_start_index
    for i in range(num_tiles // 2):
        idx = _compute_start_index(i, separation=separation)

        lefts = [op_tiles[idx] for op_tiles in arrs_tiles]
        rights = [op_tiles[idx + separation] for op_tiles in arrs_tiles]

        # Compare
        # If is_descending=True: left > right. Left gets max. We keep left.
        # If is_descending=False: right > left. Left gets min. We keep left.
        compared = _compare(lefts, rights, num_keys=num_keys, is_descending=is_descending)

        for op_idx, (kept_val, _) in enumerate(compared):
            merged_tiles[op_idx].append(kept_val)

    return tuple(merged_tiles)


def _transpose_tiles(tiles, num_rows=16):
    """Transpose tile layout from row-major to column-major (or vice versa)."""
    num_cols = len(tiles) // num_rows
    if num_cols == 0:
        return tiles
    new_tiles = []
    for c in range(num_cols):
        for r in range(num_rows):
            new_tiles.append(tiles[r * num_cols + c])
    return new_tiles


def bitonic_topk_kernel(
    in_refs,
    out_refs,
    *,
    num_keys: int,
    descending: bool,
):
    """
    Pallas kernel for bitonic top-k with k=128 in sublane format.
    """
    num_tokens = in_refs[0].shape[0]
    vocab_size = in_refs[0].shape[1]
    b = num_tokens

    if b > NUM_LANES:
        raise ValueError(f"num_tokens must be <= NUM_LANES, got {num_tokens}")

    # Process all input operands using convert_to_sublane_sort_format
    def _get_pad_val(ref):
        if descending:
            if jnp.issubdtype(ref.dtype, jnp.floating):
                return jnp.finfo(ref.dtype).min
            elif jnp.issubdtype(ref.dtype, jnp.integer):
                return jnp.iinfo(ref.dtype).min
            else:
                return -1
        return None  # Default max/nan

    arrs_tiles = tuple(
        convert_to_sublane_sort_format(
            in_ref,
            pad_val=_get_pad_val(in_ref)
        )
        for in_ref in in_refs
    )

    # Transpose tiles to Column-Major (Blocked) layout initially
    arrs_tiles = tuple(_transpose_tiles(tiles) for tiles in arrs_tiles)
    num_tiles = len(arrs_tiles[0])

    # Loop while we have more than 1 chunk (more than 16 tiles)
    while num_tiles > 16:
        # Sort chunks alternating Desc/Asc by processing chunks individually
        num_chunks = num_tiles // 16
        sorted_chunks = [[] for _ in arrs_tiles]

        for i in range(num_chunks):
            # Extract chunk i (16 tiles)
            chunk_slice = slice(i*16, (i+1)*16)
            chunk_tiles = tuple(op_tiles[chunk_slice] for op_tiles in arrs_tiles)

            # Determine direction: Even -> Desc, Odd -> Asc (assuming descending=True)
            is_chunk_desc = (i % 2 == 0) if descending else (i % 2 == 1)
            chunk_dim1_offset = 128 if is_chunk_desc else 0

            # Run sort stages 1..7 for this chunk
            for s in range(1, 8):
                chunk_tiles = _compute_subtile_substages_inner(
                    chunk_tiles,
                    num_substages=s,
                    stage=s,
                    b=b,
                    use_lane_permute=False,
                    num_keys=num_keys,
                    dim1_offset=chunk_dim1_offset,
                )

            # Append sorted chunk
            for op_idx, tiles in enumerate(chunk_tiles):
                sorted_chunks[op_idx].extend(tiles)

        arrs_tiles = tuple(sorted_chunks)

        # Merge tile pairs (C_i and C_i+1) with separation 16
        arrs_tiles = _merge_tiles_max(
            arrs_tiles,
            num_keys=num_keys,
            separation=16,
            is_descending=descending
        )

        num_tiles = len(arrs_tiles[0])

    # Final Sort of the single remaining chunk (Desc)
    final_dim1_offset = 128 if descending else 0
    for s in range(1, 8):
        arrs_tiles = _compute_subtile_substages_inner(
            arrs_tiles,
            num_substages=s,
            stage=s,
            b=b,
            use_lane_permute=False,
            num_keys=num_keys,
            dim1_offset=final_dim1_offset,
        )

    # Reconstruct.
    for tiles, out_ref in zip(arrs_tiles, out_refs):
        out = convert_from_sublane_sort_format(tiles, shape=(num_tokens, NUM_LANES))
        out_ref[...] = out


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
    def _get_pad_val(x):
        if descending:
            if jnp.issubdtype(x.dtype, jnp.floating):
                return jnp.finfo(x.dtype).min
            elif jnp.issubdtype(x.dtype, jnp.integer):
                return jnp.iinfo(x.dtype).min
            else:
                return -1
        return None

    operands = tuple(
        pad(
            x,
            block_shape=(NUM_SUBLANES, 'power_of_2_lanes'),
            prepend=(False, descending),
            val=_get_pad_val(x)
        )
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
    )(operands)[0]

    # Unpad if needed (extract k elements from the correct side)
    if not descending:
        outputs = tuple(x[:, :k] for x in outputs)
    else:
        # For descending, top-k is at the beginning after sorting
        outputs = tuple(x[:, :k] for x in outputs)

    return outputs
