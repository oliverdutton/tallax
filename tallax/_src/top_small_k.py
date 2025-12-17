"""
Top-k for small k values (k << 128) using two-stage binned approach.

Algorithm for test case (8, 10240) with k=3:
1. First binned topk: (8, 10240) -> (8, 384) using 128 bins, k=3
2. Transpose: (8, 384) -> (384, 8) using classic transpose
3. Pad: (384, 8) -> (6144, 128) where 6144 = 16*k*128, split into 48 tiles
4. Transpose: (6144, 128) -> (8, 6144)
5. Second binned topk: (8, 6144) -> (8, 24) using 8 bins, k=3
6. Transpose: (24, 8) -> (8, 24)
7. Pad: (8, 24) -> (8, 128)
8. Selection sort: For each row, find top k=3 elements
9. Write output: (8, 3)
"""

import functools
import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax._src.divide_and_filter_topk import binned_topk
from tallax._src.utils import (
    NUM_LANES,
    NUM_SUBLANES,
    log2,
    pad,
    get_dtype_info,
    to_32bit_dtype,
)


def binned_topk_axis0(
    logits,
    k: int,
    bins_topk_vals,
    bins_topk_idxs,
    num_bins: int = NUM_SUBLANES,
    unroll: int = 32,
):
    """
    Compute binned top-k across axis 0 (token dimension) instead of axis 1.

    This transposes the input, runs regular binned_topk, then transposes back.

    Args:
        logits: Input of shape [vocab_size, num_tokens] (note: transposed from usual)
        k: Number of top elements to find
        bins_topk_vals: List of k arrays for current top-k values per bin
        bins_topk_idxs: List of k arrays for current top-k indices per bin
        num_bins: Number of bins to process simultaneously
        unroll: Loop unroll factor

    Returns:
        Tuple of (bins_topk_vals, bins_topk_idxs)
    """
    # logits is already transposed to (vocab_size, num_tokens)
    # binned_topk expects (num_tokens, vocab_size), so we're good
    return binned_topk(
        logits,
        k=k,
        bins_topk_vals=bins_topk_vals,
        bins_topk_idxs=bins_topk_idxs,
        completed_k=0,
        num_bins=num_bins,
        unroll=unroll,
    )


def top_small_k_refs(
    logits_ref,
    topk_vals_ref,
    topk_idxs_ref,
    # Scratch refs
    bins_topk_vals_ref,
    bins_topk_idxs_ref,
    stage2_vals_ref,
    stage2_idxs_ref,
    final_vals_ref,
    final_idxs_ref,
    *,
    k: int,
    num_bins: int,
    bins_topm_schedule: tuple[int, ...],
):
    """
    Pallas kernel for small k top-k using two-stage binned approach.

    Shape trace for (8, 10240) with k=3:
    1. Input: (8, 10240)
    2. After binned_topk: (8, 384) where 384 = 128*3
    3. Transpose: (384, 8)
    4. Pad: (384, 8) -> (6144, 128) - dim0 to 16*k*128, dim1 to 128
    5. Transpose: (6144, 128) -> (8, 6144)
    6. After binned_topk: (8, 24) where 24 = 8*3
    7. Transpose: (24, 8) -> (8, 24)
    8. Pad dim1: (8, 24) -> (8, 128)
    9. Selection sort: Find top k=3 in each row
    10. Write output: (8, 3)
    """
    block_token, vocab_size = logits_ref.shape
    m = bins_topm_schedule[-1]

    # Initialize scratch buffers with min values for proper padding
    min_val = get_dtype_info(logits_ref).min
    min_val_32 = get_dtype_info(stage2_vals_ref).min

    bins_topk_vals_ref[...] = jnp.full(bins_topk_vals_ref.shape, min_val_32, dtype=bins_topk_vals_ref.dtype)
    bins_topk_idxs_ref[...] = jnp.full(bins_topk_idxs_ref.shape, -1, dtype=jnp.int32)
    stage2_vals_ref[...] = jnp.full(stage2_vals_ref.shape, min_val_32, dtype=stage2_vals_ref.dtype)
    stage2_idxs_ref[...] = jnp.full(stage2_idxs_ref.shape, -1, dtype=jnp.int32)
    final_vals_ref[...] = jnp.full(final_vals_ref.shape, min_val_32, dtype=final_vals_ref.dtype)
    final_idxs_ref[...] = jnp.full(final_idxs_ref.shape, -1, dtype=jnp.int32)

    # Stage 1: First binned topk with num_bins=128 bins across vocab
    # Input: (block_token, vocab_size) = (8, 10240)
    # Output: (block_token, num_bins * m) = (8, 384) for k=3
    bins_topk_vals, bins_topk_idxs = binned_topk(
        logits_ref[...],
        k=m,
        bins_topk_vals=[
            bins_topk_vals_ref[:, pl.dslice(i * num_bins, num_bins)].astype(to_32bit_dtype(logits_ref.dtype))
            for i in range(m)
        ],
        bins_topk_idxs=[
            bins_topk_idxs_ref[:, pl.dslice(i * num_bins, num_bins)]
            for i in range(m)
        ],
        num_bins=num_bins,
        completed_k=0,
        unroll=32,
    )

    # Store back to bins_topk_vals_ref: shape (block_token, num_bins * m)
    for i in range(m):
        bins_topk_vals_ref[:, pl.dslice(i * num_bins, num_bins)] = bins_topk_vals[i].astype(bins_topk_vals_ref.dtype)
        bins_topk_idxs_ref[:, pl.dslice(i * num_bins, num_bins)] = bins_topk_idxs[i]

    # Stage 2: Transpose (block_token, num_bins*m) -> (num_bins*m, block_token)
    # (8, 384) -> (384, 8)
    first_stage_size = num_bins * m
    for i in range(first_stage_size):
        for j in range(block_token):
            stage2_vals_ref[i, j] = bins_topk_vals_ref[j, i]
            stage2_idxs_ref[i, j] = bins_topk_idxs_ref[j, i]

    # Stage 3: Pad (384, 8) -> (6144, 8) where 6144 = 16*k*128
    # Padding is implicit in stage2_vals_ref allocation
    # The rest is already padded with min values from initialization

    # Stage 4: Transpose (6144, 8) -> (8, 6144) and run binned_topk
    # We need to transpose into bins_topk_vals_ref temporarily
    # Actually, we'll process this differently to use binned_topk
    # binned_topk expects (num_tokens, vocab_size)
    # So we need (block_token, padded_dim0) = (8, 6144)

    padded_dim0 = 16 * k * NUM_LANES  # 6144 for k=3

    # Transpose stage2 into bins_topk_vals_ref for second binned_topk
    # (6144, 8) -> (8, 6144)
    for i in range(block_token):
        for j in range(padded_dim0):
            bins_topk_vals_ref[i, j] = stage2_vals_ref[j, i]
            bins_topk_idxs_ref[i, j] = stage2_idxs_ref[j, i]

    # Run second binned_topk with num_bins=block_token=8 bins
    # Input: (block_token, padded_dim0) = (8, 6144)
    # Output: list of m arrays of shape (block_token, block_token) = (8, 8)
    # Combined: (8, block_token * m) = (8, 24) for k=3
    bins_topk_vals2, bins_topk_idxs2 = binned_topk(
        bins_topk_vals_ref[:, :padded_dim0],
        k=m,
        bins_topk_vals=[
            stage2_vals_ref[i * block_token:(i+1) * block_token, :block_token].astype(to_32bit_dtype(logits_ref.dtype))
            for i in range(m)
        ],
        bins_topk_idxs=[
            stage2_idxs_ref[i * block_token:(i+1) * block_token, :block_token]
            for i in range(m)
        ],
        num_bins=block_token,
        completed_k=0,
        unroll=32,
    )

    # Store to stage2_vals_ref: (block_token, block_token * m) = (8, 24)
    # We'll use the first 24 rows of stage2_vals_ref
    for i in range(m):
        for tok in range(block_token):
            for bin in range(block_token):
                stage2_vals_ref[i * block_token + bin, tok] = bins_topk_vals2[i][tok, bin].astype(stage2_vals_ref.dtype)
                stage2_idxs_ref[i * block_token + bin, tok] = bins_topk_idxs2[i][tok, bin]

    # Stage 5: Transpose (24, 8) -> (8, 24) using classic transpose
    second_stage_size = block_token * m  # 24 for k=3

    for i in range(block_token):
        for j in range(second_stage_size):
            final_vals_ref[i, j] = stage2_vals_ref[j, i]
            final_idxs_ref[i, j] = stage2_idxs_ref[j, i]

    # Stage 6: Pad (8, 24) -> (8, 128) - already done via buffer allocation

    # Stage 7: Pad dim0 to power of 2: (8, 128) -> (8, 128)
    # log2(8) = 3, so 2^3 = 8, already a power of 2
    output_dim0 = 2 ** log2(block_token)

    # Stage 8: For each token (row), find top k elements using selection sort
    # Since k is small (e.g., 3), selection sort is efficient
    # For each row in (8, 24), extract top k=3
    for ki in range(k):
        for i in range(block_token):
            # Find max in positions [ki, second_stage_size)
            max_val = final_vals_ref[i, ki]
            max_idx_pos = ki
            for j in range(ki + 1, second_stage_size):
                if final_vals_ref[i, j] > max_val:
                    max_val = final_vals_ref[i, j]
                    max_idx_pos = j

            # Swap position ki with max_idx_pos
            if max_idx_pos != ki:
                temp_val = final_vals_ref[i, ki]
                temp_idx = final_idxs_ref[i, ki]
                final_vals_ref[i, ki] = final_vals_ref[i, max_idx_pos]
                final_idxs_ref[i, ki] = final_idxs_ref[i, max_idx_pos]
                final_vals_ref[i, max_idx_pos] = temp_val
                final_idxs_ref[i, max_idx_pos] = temp_idx

    # Stage 9: Write output (top k already in positions 0:k)
    for i in range(block_token):
        for j in range(k):
            topk_vals_ref[i, j] = final_vals_ref[i, j]
            topk_idxs_ref[i, j] = final_idxs_ref[i, j]


@functools.partial(
    jit,
    static_argnames=(
        "k",
        "block_token",
        "num_bins",
        "bins_topm_schedule",
        "interpret",
    ),
)
def top_small_k(
    logits,
    k: int,
    block_token: int = NUM_SUBLANES,
    num_bins: int = NUM_LANES,
    bins_topm_schedule: tuple[int, ...] | None = None,
    interpret: bool = False,
):
    """
    Compute top-k for small k values (k << 128) using two-stage binned approach.

    This is optimized for small k (e.g., k=3) where standard approaches are inefficient.

    Algorithm:
    1. First binned topk with 128 bins to reduce vocab_size -> 128*k elements
    2. Transpose
    3. Pad to 16*k tiles
    4. Second binned topk with 8 bins across the other axis
    5. Bitonic sort the remaining elements
    6. Extract final top-k

    Args:
        logits: Input logits of shape [num_tokens, vocab_size]
        k: Number of top elements (should be small, e.g., 3)
        block_token: Number of tokens per block (default: 8)
        num_bins: Number of bins for first stage (default: 128)
        bins_topm_schedule: Schedule for binned topk stages (default: (k,))
        interpret: Whether to run in interpret mode

    Returns:
        Tuple of (topk_vals, topk_idxs):
            - topk_vals: Top-k values of shape [num_tokens, k]
            - topk_idxs: Top-k indices of shape [num_tokens, k]
    """
    if k > NUM_SUBLANES:
        raise ValueError(f"top_small_k only supports k <= {NUM_SUBLANES}, got k={k}")

    unpadded_num_tokens = logits.shape[0]

    # Pad logits to block_token multiple
    logits = pad(logits, (block_token, NUM_LANES), val='min')
    num_tokens, vocab_size = logits.shape

    # Default schedule
    if bins_topm_schedule is None:
        bins_topm_schedule = (k,)
    bins_topm_schedule = (0,) + tuple(sorted(set(bins_topm_schedule)))

    # Calculate buffer sizes
    m = bins_topm_schedule[-1]
    first_stage_size = num_bins * m  # e.g., 128 * 3 = 384
    padded_dim0 = 16 * k * NUM_LANES  # e.g., 16 * 3 * 128 = 6144
    second_stage_size = block_token * m  # e.g., 8 * 3 = 24

    # Output shapes
    output_shapes = (
        jax.ShapeDtypeStruct((num_tokens, k), logits.dtype),
        jax.ShapeDtypeStruct((num_tokens, k), jnp.int32),
    )

    # Scratch shapes
    scratch_shapes = [
        # bins_topk_vals_ref: Used for both stages
        pltpu.VMEM((block_token, max(first_stage_size, padded_dim0)), to_32bit_dtype(logits.dtype)),
        # bins_topk_idxs_ref: Used for both stages
        pltpu.VMEM((block_token, max(first_stage_size, padded_dim0)), jnp.int32),
        # stage2_vals_ref: After first transpose, pad (384,8) to (6144,128) for compressed format tiles
        pltpu.VMEM((padded_dim0, NUM_LANES), to_32bit_dtype(logits.dtype)),
        # stage2_idxs_ref: After first transpose
        pltpu.VMEM((padded_dim0, NUM_LANES), jnp.int32),
        # final_vals_ref: For final selection sort, shape (block_token, NUM_LANES)
        pltpu.VMEM((block_token, NUM_LANES), to_32bit_dtype(logits.dtype)),
        # final_idxs_ref: For final selection sort
        pltpu.VMEM((block_token, NUM_LANES), jnp.int32),
    ]

    outputs = pl.pallas_call(
        functools.partial(
            top_small_k_refs,
            k=k,
            num_bins=num_bins,
            bins_topm_schedule=bins_topm_schedule,
        ),
        in_specs=(
            pl.BlockSpec((block_token, vocab_size), lambda i: (i, 0)),
        ),
        out_shape=output_shapes,
        scratch_shapes=tuple(scratch_shapes),
        grid=(num_tokens // block_token,),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=int(0.9 * 2**27)
        ),
        interpret=interpret,
    )(logits)

    topk_vals, topk_idxs = outputs

    # Unpad
    topk_vals = topk_vals[:unpadded_num_tokens]
    topk_idxs = topk_idxs[:unpadded_num_tokens]

    return topk_vals, topk_idxs
