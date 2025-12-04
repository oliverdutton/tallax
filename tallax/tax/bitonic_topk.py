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

for (b, n) shape input
The total number of merges is log2(n // num_lanes)
The final (NUM_LANES // b) merges are intra permutations

"""

import functools
import math
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
    pad,
    canonicalize_operand,
    transpose_list_of_lists,
    convert_to_sublane_sort_format,
    convert_from_sublane_sort_format,
    to_32bit_dtype,
)
from tallax.tax.sort import (
    _compute_subtile_substages_inner,
    _compare,
    _compute_start_index,
)


def _compute_padded_shape(dim0: int, dim1: int) -> tuple[int, int]:
    """Compute padded shape for bitonic top-k.

    Requirements:
    - dim1 must be a multiple of NUM_LANES (for convert_to_sublane_sort_format)
    - dim0 * dim1 must be multiple of NUM_LANES**2 (for tiling)

    The bitonic merge handles odd tile counts via remainder propagation,
    so dim1/NUM_LANES does NOT need to be a power of 2.

    Strategy:
    - For small inputs (prod < NUM_LANES**2): pad dim0 to make prod = NUM_LANES**2
    - For larger inputs: choose between padding dim0 vs dim1 to minimize total size

    Examples:
    - (8, 256) -> (64, 256): prod = 16384 = NUM_LANES**2
    - (8, 8320) -> (8, 10240): prod = 81920 = 5 * NUM_LANES**2 (pad dim1, not dim0!)

    Args:
        dim0: First dimension size
        dim1: Second dimension size

    Returns:
        Tuple of (padded_dim0, padded_dim1)
    """
    # Step 1: Pad dim1 to next multiple of NUM_LANES (minimal padding)
    padded_dim1_min = pl.cdiv(dim1, NUM_LANES) * NUM_LANES

    # Step 2: Check if we can make product exactly NUM_LANES**2 by padding dim0 only
    prod = dim0 * padded_dim1_min
    if prod < NUM_LANES**2:
        # For small inputs, pad dim0 to make product exactly NUM_LANES**2
        padded_dim0 = NUM_LANES**2 // padded_dim1_min
        return padded_dim0, padded_dim1_min

    # Step 3: For larger inputs, choose best padding strategy
    # Option A: Pad dim0 minimally (keep dim1 = padded_dim1_min)
    gcd_val_a = math.gcd(padded_dim1_min, NUM_LANES**2)
    required_multiple_a = NUM_LANES**2 // gcd_val_a
    padded_dim0_a = pl.cdiv(dim0, required_multiple_a) * required_multiple_a
    cost_a = padded_dim0_a * padded_dim1_min

    # Option B: Pad dim1 to reduce dim0 padding requirement (keep dim0)
    gcd_val_b = math.gcd(dim0, NUM_LANES**2)
    required_multiple_b = NUM_LANES**2 // gcd_val_b
    # Ensure it's also a multiple of NUM_LANES
    required_multiple_b = pl.cdiv(required_multiple_b, NUM_LANES) * NUM_LANES
    padded_dim1_b = pl.cdiv(padded_dim1_min, required_multiple_b) * required_multiple_b
    cost_b = dim0 * padded_dim1_b

    # Choose the option with lower cost
    if cost_a <= cost_b:
        return padded_dim0_a, padded_dim1_min
    else:
        return dim0, padded_dim1_b


def _merge_max_crosstile(
    arrs_tiles, b, num_keys: int = 1
):
  """Perform crosstile comparison keeping max values.

  Args:
    arrs_tiles: Tuple of lists of tile arrays
    b: Block size (num_tokens)
    num_keys: Number of sort keys

  Returns:
    Tuple of lists with half the tiles (max halves only), plus remainder if odd
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

  # Handle odd number of tiles - add the remainder tile directly
  if num_tiles % 2 == 1:
    remainder_idx = num_tiles - 1
    for j, arr in enumerate(arrs_tiles):
      outs_tiles[j].append(arr[remainder_idx])

  return outs_tiles


def compute_bitonic_top_k_stages(arrs_tiles, num_keys, shape):
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
        shape: untransformed input shape

    Returns:
        Tuple of lists of merged tile arrays
    """
    # Target number of tiles after cross-tile merging
    # For b < NUM_LANES: we want (NUM_LANES // NUM_SUBLANES) tiles = 16 tiles
    # For b >= NUM_LANES: we want more tiles proportional to b
    b = shape[0]
    log_lanes = log2(NUM_LANES)
    num_merges = log2(shape[1] // NUM_LANES)
    num_intra_merges = min(
    log2(pl.cdiv(NUM_LANES, b)), num_merges)
    # are intra permutations

    # Build bitonic sequences up to length 64 (stage 6)
    for stage in range(1, log_lanes):  # stages 1-6 inclusive
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
    for _ in range(num_merges - num_intra_merges):
      # Run substages sorting NUM_LANES but with stage for merging bitonic sequences
      # so different tile sets have different orders.
      # tile 0 is different order to tile max(1,b//NUM_LANES), with which it will be max merged
      merge_stage = log2(NUM_LANES * max(1, NUM_LANES // b))
      arrs_tiles = _compute_subtile_substages_inner(
        arrs_tiles,
        num_substages=log_lanes,
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

    # Progressive intra-tile merging with lane 
    for i in range(num_intra_merges)[::-1]:
        distance = b * (2**i)
        # Calculate stage based on current merge size
        # Stage = log2(2 * distance * b / NUM_LANES * NUM_LANES) = log2(2 * distance)        
        arrs_tiles = _compute_subtile_substages_inner(
          arrs_tiles,
          num_substages=log_lanes,
          stage=log_lanes+i,
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

    # Final sort: convert bitonic sequence to fully descending order
    # Use dim1_offset=2**7 to ensure descending direction
    arrs_tiles = _compute_subtile_substages_inner(
        arrs_tiles,
        num_substages=log_lanes,
        stage=log_lanes,
        dim1_offset=NUM_LANES,
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
    1. Pad input to satisfy alignment requirements
    2. Convert to sublane transposed format: (num_tokens, vocab) -> (128, num_tokens*chunks)
    3. Run bitonic top-k stages to select top 128 values per token
    4. Convert back from sublane format
    5. Unpad and extract top-128 per token
    """
    shape = in_refs[0].shape

    # Compute padded shape that satisfies alignment requirements
    padded_dim0, padded_dim1 = _compute_padded_shape(shape[0], shape[1])

    # Pad both dimensions if needed
    arrs = [pad(in_ref[...], block_shape=(padded_dim0, padded_dim1))
            for in_ref in in_refs]
    arrs = [x.astype(to_32bit_dtype(x.dtype)) for x in arrs]

    # Convert to sublane transposed format
    arrs_tiles = [
        convert_to_sublane_sort_format(arr)
        for arr in arrs
    ]

    # Run bitonic top-k algorithm
    arrs_tiles = compute_bitonic_top_k_stages(arrs_tiles, num_keys=num_keys, shape=arrs[0].shape)

    # Convert back from sublane format and unpad to original shape
    for tiles, out_ref in zip(arrs_tiles, out_refs, strict=True):
        out = convert_from_sublane_sort_format(tiles, dim0=arrs[0].shape[0])[:shape[0], :NUM_LANES]
        out_ref[...] = out[:out_ref.shape[0]].astype(out_ref.dtype)


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

    Supports arbitrary input shapes - padding is handled automatically:
    - For small inputs (prod < NUM_LANES²): pads dim0 to make prod = NUM_LANES²
    - For larger inputs: pads both dims minimally to satisfy alignment

    Args:
        operand: Input array(s) of shape [num_tokens, vocab_size].
                Can be a single array or sequence of arrays.
                Any vocab_size is supported (will be padded automatically).
        k: Number of top elements (must be NUM_LANES=128).
        num_keys: Number of arrays to use as sort keys.
        descending: If True, sort in descending order (default for top-k).
        interpret: If True, run in CPU interpret mode.

    Returns:
        Tuple of arrays (same length as input operands):
            - Each array has shape [num_tokens, k]

    Raises:
        ValueError: If k != NUM_LANES
    """
    if k != NUM_LANES:
        raise ValueError(
            f"bitonic_topk only supports k=NUM_LANES={NUM_LANES}, got k={k}"
        )

    operands, shape = canonicalize_operand(operand)
    num_tokens, vocab_size = shape
    # Define output shapes
    output_shapes = [
        jax.ShapeDtypeStruct((num_tokens, NUM_LANES), op.dtype)
        for op in operands
    ]
    outputs = pl.pallas_call(
        functools.partial(
            bitonic_topk_kernel,
            num_keys=num_keys,
            descending=descending,
        ),
        #in_specs=(tuple(pl.BlockSpec() for _ in operands),),
        out_shape=(output_shapes,),
        
        #out_specs=([pl.BlockSpec() for _ in output_shapes),),
        grid=(),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=int(0.9 * 2**27)
        ),
        interpret=interpret,
    )(operands)[0]
    return tuple(x[:, :k] for x in outputs)