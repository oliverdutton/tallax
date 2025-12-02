"""
Bitonic Top-K for k=NUM_LANES=128 using sublane transposed format.

This implementation is optimized for TPU with k=128 and works entirely in
sublane transposed format to maximize efficiency of permutation operations.

ISSUES WITH THIS CODE:
======================

CRITICAL SYNTAX ERRORS:
1. Missing commas in multiple _compute_subtile_substages_inner calls (lines ~37, ~50, ~58, ~67, ~92, ~104)
2. Variable 'separation' undefined in _merge_max_crosstile function
3. Variable 'shape' undefined in permutation creation (line ~109)

LOGIC ERRORS:
4. len(arrs_tiles) should be len(arrs_tiles[0]) - arrs_tiles is tuple of lists (line ~52)
5. log2(b/NUM_LANES) produces negative values when b < NUM_LANES (lines ~33, ~52, ~90)
6. convert_to_sublane_sort_format doesn't accept pad_val parameter (line ~133)
7. Separation calculation is incorrect - should be pl.cdiv(b, NUM_LANES) not log2-based
8. Stage numbering logic is unclear and likely incorrect

ALGORITHM ISSUES:
9. Initial loop runs stages 1-6, but comment says 1-7
10. While loop termination condition uses negative log2 value
11. Progressive merging section incomplete - permutation over undefined shape
12. Crosstile merge doesn't properly reduce tile count
13. The overall flow from building bitonic sequences to merging is unclear

See inline comments marked with "# ISSUE #X:" for details.

Run stages 1,2,3,4,5,6,7 and have length 128 bitonic sequences.
crosstile compare with separation of 16 * pl.cdiv(b,128)
b must be power of 2
throw away the lower tiles from that.

now we have bitonic 128 values.
run substages 6,5,4,3,2,1,0. Stage 7+log2(128/b) [possible negative value]
Now we have 16 * pl.cdiv(b,128) tiles left. we must combine intra tile

these must be (b,128) in transposed format in a bitonic form?
run substages 6,5,4,3,2,1,0. Stage 7

end of stage 1, 2**1=length 2 sorted
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
  """Perform substage by comparing explicit tile pairs."""
  # ISSUE #2: 'separation' is used below but never defined or passed as parameter
  num_tiles = len(arrs_tiles[0])
  outs_tiles = [[] for t in arrs_tiles]
  for i in range(num_tiles // 2):
    idx = _compute_start_index(i, separation=pl.cdiv(b, NUM_LANES))
    lefts, rights = (
        transpose_list_of_lists(arrs_tiles)[j]
        for j in (idx, idx + separation)  # ISSUE #2: separation undefined here!
    )
    for i, (o_left, _) in enumerate(_compare(
        lefts, rights, is_descending=True, num_keys=num_keys
    )):
      outs_tiles[i].append(o_left)
  return outs_tiles


def compute_bitonic_top_k_stages(arrs_tiles, num_keys, b):
    """
    Progressive lane permute merging with decreasing distances and stages.

    Runs log2(128//b) iterations with:
    - Iteration i: distance = 64 >> i, stage = initial_stage - i
    - Each iteration: permute, max merge, run substages 6-0

    Args:
        arrs_tiles: Tuple of lists of tile arrays
        initial_stage: Starting stage (7 + log2(128//b))
        num_keys: Number of sort keys
        b: Block size (num_tokens)

    Returns:
        Tuple of lists of merged tile arrays
    """

    # ISSUE #5: log2(b/NUM_LANES) is negative when b < NUM_LANES
    # Example: b=8, NUM_LANES=128 -> log2(8/128) = log2(0.0625) ≈ -4
    # This makes separation negative which doesn't make sense!
    separation = (NUM_LANES // NUM_SUBLANES) * log2(b/NUM_LANES)

    # ISSUE #9: Comment says "run stages 1,2,3,4,5,6,7" but loop only runs 1-6
    for stage in range(1, 7):
      # sort up to  bitonic
      arrs_tiles = _compute_subtile_substages_inner(
        arrs_tiles,
        num_substages=stage,
        stage=stage,
        dim1_offset=0  # ISSUE #1: Missing comma here!
        b=b,
        num_keys=num_keys,
        use_lane_permute=False,
      )

    # ISSUE #4: len(arrs_tiles) is wrong - arrs_tiles is a tuple, so len returns number of operands
    # Should be len(arrs_tiles[0]) to get number of tiles
    # ISSUE #5: log2(b/NUM_LANES) is negative when b < 128
    # ISSUE #10: This condition will almost never work as intended
    while len(arrs_tiles) > (NUM_LANES // NUM_SUBLANES) * log2(b/NUM_LANES):
      arrs_tiles = _compute_subtile_substages_inner(
        arrs_tiles,
        num_substages=7,
        stage=7+log2(NUM_LANES/b),
        dim1_offset=0,  # ISSUE #1: Missing comma
        b=b,
        num_keys=num_keys,
        use_lane_permute=False,
      )
      # now we have 128 length sort, but with different order between each set of 16 tiles
      arrs_tiles = _merge_max_crosstile(
          arrs_tiles, b=b,
          num_keys=num_keys
      )

    # Progressive merging: log2(128//b) iterations
    num_iterations = log2(NUM_LANES // b)

    for i in range(num_iterations):
      #current_stage = initial_stage - i
      # 8, so 0000128128128...15*128 so we want stage
      # ISSUE #8: Stage calculation unclear, num_iterations-1-i produces descending values
      stage = 7+num_iterations-1-i

      arrs_tiles = _compute_subtile_substages_inner(
        arrs_tiles,
        num_substages=7,
        stage=stage,
        dim1_offset=0,  # ISSUE #1: Missing comma
        b=b,
        num_keys=num_keys,
        use_lane_permute=False,
      )

      distance = NUM_LANES >> (i+1)  # 64, 32, 16, ..., down to b

      # Create permutation: XOR with distance (equivalent to roll for power-of-2 distances)
      # Element at position i gets combined with element at i XOR distance
      # We need indices matching the full array shape (128, N)
      # ISSUE #3: 'shape' is not defined anywhere!
      index = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
      permutation = jnp.bitwise_xor(index, distance)
      # Permute using take_along_axis (TPU-supported for (8, 128) tiles)
      arrs_tiles_permuted = jax.tree.map(
        lambda tile: jnp.take_along_axis(tile, permutation, axis=1), arrs_tiles
      )
      outs_tiles = [[] for _ in arrs_tiles]
      for _, (lefts, rights) in enumerate(zip(
            *map(transpose_list_of_lists, (arrs_tiles, arrs_tiles_permuted)),
            strict=True
        )):
          for i, (o,_) in enumerate(_compare(lefts, rights,
                                         is_descending=True,
                                         num_keys=num_keys)):
            outs_tiles[i].append(o)
      arrs_tiles = outs_tiles

    # final (b,128) bitonic sort to descending
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

    # ISSUE #6: convert_to_sublane_sort_format doesn't accept pad_val parameter
    # See utils.py:262 - function signature is: def convert_to_sublane_sort_format(arr)
    arrs_tiles = tuple(
        convert_to_sublane_sort_format(
            in_ref[...].astype(jnp.float32),
            pad_val=_get_pad_val(in_ref)  # This parameter doesn't exist!
        )
        for in_ref in in_refs
    )

    arrs_tiles = compute_bitonic_top_k_stages(arrs_tiles, num_keys=1, b=b)
    # Extract results and convert back from sublane format
    # Use convert_from_sublane_sort_format to reconstruct the (num_tokens, 128) array
    for tiles, out_ref in zip(arrs_tiles, out_refs):
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

    operands = tuple(
        pad(
            x,
            block_shape=(NUM_SUBLANES, 'power_of_2_lanes'),
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
