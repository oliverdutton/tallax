"""
Bitonic Top-K for k=NUM_LANES=128 using sublane transposed format.

This implementation is optimized for TPU with k=128 and works entirely in
sublane transposed format to maximize efficiency of permutation operations.

Algorithm:
- Convert input to sublane transposed format: (num_tokens, vocab) -> (128, num_tokens*chunks)
- Build bitonic sequences using stages 1-7
- Cross-tile merge with max selection, reducing tile count
- Progressive lane permute merging with decreasing distances
- Convert back to original format

For input (b, vocab) where b=num_tokens:
- After sublane transpose: (128, b*n) split into (8, 128) tiles
- Number of tiles: (b * n_vocab_chunks) // 8 where n_vocab_chunks = vocab // 128
- Target: reduce to (NUM_LANES // NUM_SUBLANES) * max(1, b // NUM_LANES) tiles
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

def _merge_max_crosstile(
    arrs_tiles, b, num_keys: int = 1
):
  """Perform crosstile comparison keeping max values.

  Args:
    arrs_tiles: Tuple of lists of tile arrays
    b: Block size (num_tokens)
    num_keys: Number of sort keys

  Returns:
    Tuple of lists with half the tiles (max halves only)
  """
  num_tiles = len(arrs_tiles[0])
  separation = max(1, b // NUM_LANES)  # Tiles per token block row
  outs_tiles = [[] for t in arrs_tiles]

  for i in range(num_tiles // 2):
    idx = _compute_start_index(i, separation=separation)
    lefts, rights = (
        transpose_list_of_lists(arrs_tiles)[j]
        for j in (idx, idx + separation)
    )
    # Keep only max (left) values, discard min (right)
    for j, (o_left, _) in enumerate(_compare(
        lefts, rights, is_descending=True, num_keys=num_keys
    )):
      outs_tiles[j].append(o_left)
  return outs_tiles


def compute_bitonic_top_k_stages(arrs_tiles, num_keys, b, dim1_size):
    """
    Progressive bitonic merge for top-k selection.

    Strategy:
    1. Build bitonic sequences (stages 1-6) within tiles
    2. Cross-tile bitonic merge until we reach target tile count
    3. Final progressive merge with lane permutations
    4. Sort final bitonic sequence to descending order

    Args:
        arrs_tiles: Tuple of lists of tile arrays
        num_keys: Number of sort keys
        b: Block size (num_tokens, must be power of 2)
        dim1_size: Size of second dimension after sublane transpose

    Returns:
        Tuple of lists of merged tile arrays
    """
    # Target number of tiles after cross-tile merging
    # For b < NUM_LANES: we want (NUM_LANES // NUM_SUBLANES) tiles = 16 tiles
    # For b >= NUM_LANES: we want more tiles proportional to b
    target_num_tiles = (NUM_LANES // NUM_SUBLANES) * max(1, b // NUM_LANES)

    # Build bitonic sequences up to length 64 (stage 6)
    for stage in range(1, 7):  # stages 1-6 inclusive
      arrs_tiles = _compute_subtile_substages_inner(
        arrs_tiles,
        num_substages=stage,
        stage=stage,
        dim1_offset=0,
        b=b,
        num_keys=num_keys,
        use_lane_permute=False,
      )

    # Cross-tile merging: reduce tile count by half each iteration
    # Keep merging until we hit target tile count
    while len(arrs_tiles[0]) > target_num_tiles:
      # Run substages sorting NUM_LANES but with stage for merging bitonic sequences
      # so different tile sets have different orders.
      # tile 0 is different order to tile max(1,b//NUM_LANES), with which it will be max merged
      merge_stage = log2(NUM_LANES * max(1, NUM_LANES // b))
      arrs_tiles = _compute_subtile_substages_inner(
        arrs_tiles,
        num_substages=7,
        stage=merge_stage,
        dim1_offset=0,
        b=b,
        num_keys=num_keys,
        use_lane_permute=False,
      )

      # Cross-tile comparison: keep max half, discard min half
      arrs_tiles = _merge_max_crosstile(
          arrs_tiles,
          b=b,
          num_keys=num_keys
      )

    # Progressive intra-tile merging with lane permutations
    distance = min(NUM_LANES // 2, (dim1_size * b) // NUM_LANES)
    while distance >= b:
        # Calculate stage based on current merge size
        # Stage = log2(2 * distance * b / NUM_LANES * NUM_LANES) = log2(2 * distance)
        stage = log2(2 * distance)
        arrs_tiles = _compute_subtile_substages_inner(
          arrs_tiles,
          num_substages=7,
          stage=stage,
          dim1_offset=0,
          b=b,
          num_keys=num_keys,
          use_lane_permute=False,
        )

        # Create permutation indices for tiles using iota_tile
        permutation = jnp.bitwise_xor(iota_tile(1), distance)

        # Apply permutation to all tiles
        arrs_tiles_permuted = jax.tree.map(
          lambda tile: jnp.take_along_axis(tile, permutation, axis=1),
          arrs_tiles
        )

        # Compare and merge with permuted values
        outs_tiles = [[] for _ in arrs_tiles]
        for _, (lefts, rights) in enumerate(zip(
              *map(transpose_list_of_lists, (arrs_tiles, arrs_tiles_permuted)),
              strict=True
          )):
            for j, (o, _) in enumerate(_compare(
                lefts, rights,
                is_descending=True,
                num_keys=num_keys
            )):
              outs_tiles[j].append(o)
        arrs_tiles = outs_tiles

        # Halve the distance for next iteration
        distance = distance >> 1

    # Final sort: convert bitonic sequence to fully descending order
    # Use dim1_offset=2**7 to ensure descending direction
    arrs_tiles = _compute_subtile_substages_inner(
        arrs_tiles,
        num_substages=7,
        stage=7,
        dim1_offset=2**7,
        b=b,
        num_keys=num_keys,
        use_lane_permute=False,
    )
    return arrs_tiles


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
    2. Run bitonic top-k stages to select top 128 values per token
    3. Convert back from sublane format
    4. Extract top-128 per token
    """
    num_tokens = in_refs[0].shape[0]
    vocab_size = in_refs[0].shape[1]
    b = num_tokens

    if b > NUM_LANES:
        raise ValueError(f"num_tokens must be <= NUM_LANES, got {num_tokens}")

    # Convert to sublane transposed format
    # Note: padding handled by convert_to_sublane_sort_format internally
    arrs_tiles = tuple(
        convert_to_sublane_sort_format(in_ref[...].astype(jnp.float32))
        for in_ref in in_refs
    )

    # Calculate dim1_size from number of tiles
    # After sublane transpose: (128, dim1_size) split into tiles of (8, 128)
    # num_tiles = 16 * (dim1_size / 128), so dim1_size = num_tiles * 8
    num_tiles = len(arrs_tiles[0])
    dim1_size = num_tiles * NUM_SUBLANES

    # Run bitonic top-k algorithm
    arrs_tiles = compute_bitonic_top_k_stages(arrs_tiles, num_keys=num_keys, b=b, dim1_size=dim1_size)

    # Convert back from sublane format and write to output
    for tiles, out_ref in zip(arrs_tiles, out_refs, strict=True):
        out = convert_from_sublane_sort_format(tiles, shape=(num_tokens, NUM_LANES))
        out_ref[...] = out.astype(out_ref.dtype)


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

    # Critical: ensure vocab is large enough that convert_to_sublane_sort_format
    # won't need to pad (which would use wrong default values).
    # After transpose in sublane format: (128, b*n) where n = vocab//128
    # We need b*n >= 128, so vocab >= 128*128/b = 16384/b
    min_vocab_for_no_internal_padding = (NUM_LANES * NUM_LANES) // num_tokens

    operands = tuple(
        pad(
            x,
            block_shape=(NUM_SUBLANES, max(NUM_LANES, min_vocab_for_no_internal_padding)),
            val=_get_pad_val(x)
        )
        for x in operands
    )

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
            pl.BlockSpec()
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
    return tuple(x[:num_tokens, :k] for x in outputs)