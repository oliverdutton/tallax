"""
Bitonic Top-K for k<=NUM_LANES (128) using compressed transpose format.

This implementation is optimized for TPU and works entirely in
compressed transpose format to maximize efficiency of permutation operations.

Algorithm:
- Convert input to compressed transpose format: (num_tokens, vocab) -> (NUM_LANES, num_tokens*chunks)
- Build bitonic sequences using stages 1-6 (so sorted in 64 length chunks)
- Cross-tile merge with max selection, reducing tile count
- Progressive sublane permute merging with decreasing distances
- Convert back to original format


Before
this is better than before when substages were all compiled (num_stages - substage) times. so for 2**15=32768 thats 15,14,13,12,11,10,9,8,7,6,5,4,3,2,1 substages = 120 times
- 42+6=48 permute compilations
- 72 tile compilations

After
This version does (num sublane stages, num tile stages, num lane stages)
there are 1+2+3=6 sublane compilations in the stage unrolled first 3 stages
the tile substages get compiled 2 times
the lane substages get compiled once

so for (16, 32768). 6 sublane, 9 tile, 3 lane. 
- 9 permute compilations
- 18 tile compilations

Is the permute compilation dominating? - add skip_permutes kwarg and check T/F

the issue is still linear compile and trace times due to number of tiles


"""


import functools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import jit
#from jax import make_jaxpr
#from jax.extend.core import jaxpr_as_fun, ClosedJaxpr
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax._src.utils import (
    NUM_LANES,
    NUM_SUBLANES,
    log2,
    flatten,
    ceil_multiple,
    iota_tile,
    pad,
    canonicalize_operand,
    transpose_list_of_lists,
    to_compressed_transpose_format,
    from_compressed_transpose_format,
    to_32bit_dtype,
    join_tiles_to_array,
    split_array_to_tiles,
    create_bit_indicator,
)
#from tallax._src.cse import cse_until_fixpoint
from tallax._src.sort import (
    compare_and_swap,
    compute_pair_slice_start_index,
)

CONCAT_TILES = True

def _compute_padded_shape(unpadded_dim0: int, unpadded_dim1: int, k: int) -> tuple[int, int]:
  """Compute padded shape compatible with compressed transpose format requirements.

  This function finds the minimal padded shape that satisfies the constraints:
  - dim0 is a power of 2 between NUM_SUBLANES and NUM_LANES (inclusive)
  - dim1 is a multiple of k
  - num_elems must be divisible by NUM_LANES^2 so mosaic lowers the split and
    concat on full tiles, subtile concat not supported

  Args:
    unpadded_dim0: Original first dimension size
    unpadded_dim1: Original second dimension size
    k: Target top-k size (must be power of 2 for padding calculation purposes)

  Returns:
    Tuple of (padded_dim0, padded_dim1) compatible with compressed transpose format
  """
  if unpadded_dim0 >= NUM_LANES:
    dim0 = ceil_multiple(unpadded_dim0, NUM_LANES)
    dim1 = ceil_multiple(unpadded_dim1, max(k, NUM_SUBLANES))
    return (dim0, dim1)

  dim0s = [2**i for i in range(log2(NUM_SUBLANES), log2(NUM_LANES)+1)
    if 2**i >= unpadded_dim0]
  shapes = [
    (dim0, ceil_multiple(unpadded_dim1,
      NUM_LANES * NUM_LANES // dim0))
    for dim0 in dim0s]
  # take minimal num elements, larger dim0 on ties as cross tile ops are faster than cross lane
  return sorted(shapes, key=lambda x: (x[0] * x[1], -x[0]))[0]

### Bitonic Sort Implementation

def _resplit(operands, target_tile_dim0: int):    
    def _resplit_inner(operand):
      tiles = jax.tree.leaves(operand)
      dim0 = tiles[0].shape[0]
      if dim0 == target_tile_dim0:
        return tiles
      elif dim0 > target_tile_dim0:
        return flatten([jnp.split(tile, dim0//target_tile_dim0, axis=0) for tile in tiles])
      else:
        l = target_tile_dim0 // dim0
        return [jnp.concatenate(operand[i*l:(i+1)*l], axis=0) for i in range(len(tiles)//l)]
  
    return [_resplit_inner(x) for x in operands]
    
def _rejoin(operands):
  def _inner(operand):
    tiles = jax.tree.leaves(operand)
    return jnp.concatenate(tiles, axis=0)
  return [_inner(x) for x in operands]
       
   
def _compute_is_descending(stage, tile_start_offset, tile_local_offset, sort_dim_offset, compression_length):
    #assert isinstance(sort_dim_offset, int)
    
    # is_descending repeats every 2**(stage+1)
    # if the offset divides cleanly it's 0's allowing for CSE in the add of zeros (and hopefully remove of add 0)
    #sort_dim_offset %= (2**(stage+1))
    
    # unoptimized is_descending from fully computing indices
    is_descending = create_bit_indicator(stage, tile_start_offset + tile_local_offset + sort_dim_offset)
    #return is_descending
    
    if type(stage) == int:
      stage_lb = stage
      stage_ub = stage
    elif hasattr(stage, 'lower_bound') and hasattr(stage, 'upper_bound'):
      # both bounds are inclusive
      stage_lb = stage.lower_bound
      stage_ub = stage.upper_bound
    else:
      # can't optimize
      return is_descending
      
    if (stage_ub < log2(NUM_SUBLANES)) or (stage_lb >= log2(compression_length)):
        # Bit only set by iota_tile(0), same pattern for all tiles
        return create_bit_indicator(stage, tile_local_offset + sort_dim_offset)
    elif (stage_lb >= log2(NUM_SUBLANES)) and (stage_ub < log2(compression_length)):
        # Bit set by tile_offset, constant within tile, differs across tiles
        return create_bit_indicator(stage, tile_start_offset + sort_dim_offset)
    # can't optimize
    return is_descending
    

def _bitonic_sort_substage(arrs_tiles, *, substage, stage, num_keys: int, batch_size: int, sort_dim_offset: int = 0, compression_length=None):
    """Perform intra-tile bitonic comparison for sort.

    Args:
      arrs_tiles: Tuple of lists of tile arrays
      axis: Axis along which to apply permutation (0 or 1)
      separation: Distance between elements to compare within tile
      stage: Current sorting stage
      num_keys: Number of sort keys
      sort_dim_offset: Offset for bitonic order calculation
      batch_size: Batch size for computing tile offsets

    Returns:
      Tuple of lists of tiles with updated values
    """
    separation = 2**substage
    # if still arrays, we make it into one big tile so its sanitized to list[list[jax.ndarray]]
    arrs_tiles = list(map(jax.tree.leaves, arrs_tiles))

    if compression_length is None:
      compression_length = len(arrs_tiles[0]) * arrs_tiles[0][0].shape[0]
    if separation < NUM_SUBLANES or separation >= compression_length:
      # we need to permute within tiles
      axis = int(separation >= compression_length)
      intra_tile_separation = separation if axis==0 else ((separation * batch_size) // compression_length)
      
      # we need hardware tiles to lower the permute
      arrs_tiles = _resplit(arrs_tiles, NUM_SUBLANES)
      # Compute is_descending for each tile based on bitonic pattern
      tile_local_offset = iota_tile(0) + (iota_tile(1) // batch_size) * compression_length
      is_right_half = create_bit_indicator(log2(intra_tile_separation), iota_tile(axis))
      permutation = jnp.bitwise_xor(iota_tile(axis), intra_tile_separation)
      # Apply permutation to all tiles
      arrs_tiles_permuted = jax.tree.map(
        lambda tile: jnp.take_along_axis(tile, permutation, axis=axis),
        arrs_tiles
      )
  
      # Compare and merge with permuted values
      outs_tiles = [[None for _ in t] for t in arrs_tiles]
      for idx, (lefts, rights) in enumerate(zip(
            *map(transpose_list_of_lists, (arrs_tiles, arrs_tiles_permuted)),
            strict=True
        )):
          for arr_idx, out in enumerate(compare_and_swap(
            lefts, rights,
            is_descending=      _compute_is_descending(
              stage=stage, 
              tile_start_offset=idx*NUM_SUBLANES, 
              tile_local_offset=tile_local_offset, 
              sort_dim_offset=sort_dim_offset, 
              compression_length=compression_length
            ),
            is_right_half=is_right_half,
            num_keys=num_keys
          )):
            outs_tiles[arr_idx][idx] = out
    else:
      # Comparison between tiles
      arrs_tiles = _resplit(arrs_tiles, separation if CONCAT_TILES else NUM_SUBLANES)
      tile_shape = arrs_tiles[0][0].shape      
      num_tiles = len(arrs_tiles[0])
      tile_separation = separation // tile_shape[0]
      
      tile_local_offset = iota_tile(0, tile_shape) + (iota_tile(1, tile_shape) // batch_size) * compression_length

      outs_tiles = [[None for _ in t] for t in arrs_tiles]
      for i in range(num_tiles // 2):
        idx = compute_pair_slice_start_index(i, separation=tile_separation)
        lefts, rights = (transpose_list_of_lists(arrs_tiles)[j] for j in (idx, idx + tile_separation))
        for arr_idx, (out_left, out_right) in enumerate(compare_and_swap(
            lefts, rights, is_descending=_compute_is_descending(
              stage=stage, 
              tile_start_offset=idx*tile_shape[0], 
              tile_local_offset=tile_local_offset, 
              sort_dim_offset=sort_dim_offset, 
              compression_length=compression_length
            ),
            num_keys=num_keys
        )):
          outs_tiles[arr_idx][idx] = out_left
          outs_tiles[arr_idx][idx + tile_separation] = out_right
    assert all(not any([v is None for v in out_tiles]) for out_tiles in outs_tiles)
    return outs_tiles


def _bitonic_sort_substage_refs(transpose_refs, *, substages, stages, num_keys: int, batch_size: int, sort_dim_offset: int = 0, compression_length=None, slice_size=None):
  # no compression or not compatible
  if slice_size is None:
    slice_size = compression_length  
  
  sharded = tuple(2**substage < slice_size for substage in substages)
  if all(sharded):
    pass
  elif all((not b for b in sharded)):
    slice_size = compression_length
  else:
    # will switch between them. We do the longest run we can of same slice_size
    split_i = next(i for i, v in enumerate(sharded) if v!=sharded[0])
    _bitonic_sort_substage_refs(
          transpose_refs, substages=substages[:split_i], stages=stages[:split_i], num_keys=num_keys, batch_size=batch_size, sort_dim_offset=sort_dim_offset, compression_length=compression_length,  slice_size=slice_size)
    _bitonic_sort_substage_refs(
          transpose_refs, substages=substages[split_i:], stages=stages[split_i:], num_keys=num_keys, batch_size=batch_size, sort_dim_offset=sort_dim_offset, compression_length=compression_length,  slice_size=slice_size)
    
  grid_size = compression_length // slice_size
            
  @pl.loop(0, grid_size)
  def process_block(i):
    #i = SymInt(i) # to track divisors of i for is_descending optimizations
    arrs_tiles = [
        ref[pl.dslice(i * slice_size, slice_size)]
        for ref in transpose_refs
    ]
    for substage, stage in zip(substages, stages, strict=True):
      arrs_tiles = _bitonic_sort_substage(arrs_tiles, substage=substage, stage=stage, num_keys=num_keys, batch_size=batch_size, sort_dim_offset=sort_dim_offset + i * slice_size, compression_length=compression_length)
    # Write back to refs
    for ref, arr in zip(transpose_refs, _rejoin(arrs_tiles), strict=True):
      ref[pl.dslice(i * slice_size, slice_size)] = arr

'''                
class BoundedInt(jax.ndarray):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
  add lower_bound and upper_bound and tracking of what its a multiple of so % can produce 0 in some cases
'''


def bitonic_sort_arrays(operands: list[jax.Array], num_keys: int = 1, axis: int = 1, descending: bool = False, stage_unroll: int | None = None, tile_unroll: int | None = None, transpose_refs=None):
    """
    Bitonic sort using compressed transpose format with full tile unrolling.

    Similar to bitonic_topk_arrays but performs full sort without reduction.
    Uses the same tiling strategy and format conversion for efficient TPU execution.

    Handles arbitrary sort dimensions efficiently:
    - Dimensions ≤ NUM_LANES (128): Uses compressed transpose format substages
    - Dimensions > NUM_LANES: Extends with cross-lane permutation substages
    - Example: (8, 2048) sorted using stage-based bitonic reduce with full tile unrolling

    Args:
        operands: List of JAX arrays of shape (dim0, dim1)
        num_keys: Number of sort keys (default: 1)
        axis: Axis along which to perform sort (0 or 1)
        descending: If True, sort in descending order

    Returns:
        List of JAX arrays of same shape as input, sorted along specified axis
    """
    if stage_unroll is not None:
      assert transpose_refs is not None
  
    batch_axis = 1 - axis
    shape = operands[0].shape
    #unpadded_sort_dim = shape[axis]

    if axis == 1:
        padded_shape = _compute_padded_shape(shape[0], shape[1], k=NUM_SUBLANES)
    elif axis == 0:
        padded_shape = (
          ceil_multiple(shape[0], NUM_SUBLANES),
          ceil_multiple(shape[1], NUM_LANES)
        )
    else:
        raise ValueError
        
    print(f'Padding {shape} to {padded_shape}')

    # Pad both dimensions if needed
    # For ascending sort, pad with 'max' so padding values sort to the end
    # For descending sort, pad with 'min' so padding values sort to the end
    arrs = [pad(op, block_shape=padded_shape, val='min' if descending else 'max') for op in operands]
    arrs = [x.astype(to_32bit_dtype(x.dtype)) for x in arrs]
    
    num_stages = log2(shape[axis])
    stage_unroll = min(stage_unroll, num_stages) if stage_unroll is not None else num_stages
  
    def _sort_arrays(arrs):
      batch_size = arrs[0].shape[batch_axis]
      assert batch_size <= NUM_LANES

      # Convert to compressed transpose format
      arrs_tiles = jax.tree.map((to_compressed_transpose_format if axis==1 else split_array_to_tiles), arrs)

      # Offset to control ascending vs descending final order
      sort_dim_offset = int(descending) * (2**num_stages)

      # Run all bitonic sort stages
      compression_length = arrs_tiles[0].shape[0] 
      slice_size = min(
        max(tile_unroll * NUM_SUBLANES, 2**stage_unroll), compression_length) if tile_unroll is not None else compression_length
      
      '''
      for stage in range(1, num_stages + 1):
        for substage in range(stage)[::-1]:
          arrs_tiles = _bitonic_sort_substage(arrs_tiles, substage=substage, stage=stage, num_keys=num_keys, batch_size=batch_size, sort_dim_offset=sort_dim_offset, compression_length = compression_length)
      '''
      
      slice_size = min(max(2**stage_unroll, NUM_SUBLANES), compression_length)
      out_arrs_tiles = []
      for i, arrs_slice_tiles in enumerate(transpose_list_of_lists(_resplit(arrs_tiles, slice_size))):
        for stage in range(1, stage_unroll + 1):
          for substage in range(stage)[::-1]:
            arrs_slice_tiles = _bitonic_sort_substage(arrs_slice_tiles, substage=substage, stage=stage, num_keys=num_keys, batch_size=batch_size, sort_dim_offset=sort_dim_offset+i*slice_size, compression_length = compression_length)    
        out_arrs_tiles.append([jnp.concat(x, axis=0) for x in arrs_slice_tiles])
      arrs_tiles = transpose_list_of_lists(out_arrs_tiles)                    
      
      for stage in range(stage_unroll + 1, num_stages + 1):
        for substage in range(stage_unroll, stage)[::-1]:
          arrs_tiles = _bitonic_sort_substage(arrs_tiles, substage=substage, stage=stage, num_keys=num_keys, batch_size=batch_size, sort_dim_offset=sort_dim_offset, compression_length=compression_length)            
                          
        out_arrs_tiles = []
        for i, arrs_slice_tiles in enumerate(transpose_list_of_lists(_resplit(arrs_tiles, slice_size))):
          for substage in range(stage_unroll)[::-1]:
            arrs_slice_tiles = _bitonic_sort_substage(arrs_slice_tiles, substage=substage, stage=stage, num_keys=num_keys, batch_size=batch_size, sort_dim_offset=sort_dim_offset+i*slice_size, compression_length = compression_length)
          out_arrs_tiles.append([jnp.concat(x, axis=0) for x in arrs_slice_tiles])
        arrs_tiles = transpose_list_of_lists(out_arrs_tiles)                    
      
      '''    
      else:
        # enter ref usage
        for ref, arr in zip(transpose_refs, _rejoin(arrs_tiles), strict=True):
          ref[...] = arr
          
        num_crosslane_stages = log2(NUM_LANES // batch_size)
        stage_sections = (stage_unroll, num_stages - num_crosslane_stages, num_stages)
        # stages are 1-indexed
        stage_sections = tuple(i+1 for i in stage_sections)
                
        stages, substages = [],[]
        for stage in range(1, stage_sections[0]):
          for substage in range(stage)[::-1]:
            stages.append(stage)
            substages.append(substage)
        _bitonic_sort_substage_refs(
          transpose_refs, substages=substages, stages=stages, num_keys=num_keys, batch_size=batch_size, sort_dim_offset=sort_dim_offset, compression_length = compression_length,  slice_size=slice_size)
        
        for stage_lb, stage_ub in zip(stage_sections, stage_sections[1:]):
          # run the cross tile and cross lane fori_loops separately so we can make optimizations on is_descending
          @pl.loop(stage_lb, stage_ub)
          def run_dynamic_stage(stage):
            
            # bounds are inclusive on both ends
            # this is used to make optimizations on is_descending inside the code
            #stage = SymInt(stage, lower_bound=stage_lb, upper_bound=stage_ub-1)
            
            for substage in range(stage_lb, stage_ub)[::-1]:
              @pl.when(stage > substage)
              def run_substage():
                _bitonic_sort_substage_refs(
                  transpose_refs, substages=(substage,), stages=(stage,), num_keys=num_keys, batch_size=batch_size, sort_dim_offset=sort_dim_offset, compression_length = compression_length,  slice_size=slice_size)
          
          substages = tuple(range(stage_lb)[::-1])
          stages = (stage,)*len(substages)
          _bitonic_sort_substage_refs(
          transpose_refs, substages=substages, stages=stages, num_keys=num_keys, batch_size=batch_size, sort_dim_offset=sort_dim_offset, compression_length = compression_length,  slice_size=slice_size)
        # back in array flow
        arrs_tiles = [[ref[...]] for ref in transpose_refs]
      '''
      # Convert back from compressed transpose format
      if axis == 1:
        arrs = [from_compressed_transpose_format(tiles, dim0=batch_size) for tiles in arrs_tiles]
      else:
        arrs = [join_tiles_to_array(tiles, dim0=ceil_multiple(2**num_stages, NUM_SUBLANES)) for tiles in arrs_tiles]
      return arrs

    # wrapping to act on batch_size <= NUM_LANES in the kernel
    arrs = [
      jnp.concatenate(arr_slices, axis=batch_axis)
      for arr_slices in transpose_list_of_lists(
        [_sort_arrays(arrs)
        for arrs in transpose_list_of_lists([
        jnp.split(arr, pl.cdiv(padded_shape[batch_axis], NUM_LANES), axis=batch_axis) for arr in arrs])
    ])]
    # Unpad to original shape
    return [arr[:shape[0], :shape[1]] for arr in arrs]


def bitonic_sort_refs(
    in_refs,
    out_refs,
    *,
    num_keys: int,
    descending: bool,
    tile_unroll: int | None = None,
    stage_unroll: int | None = None,
):
    """
    Pallas kernel for bitonic sort in compressed transpose format.

    Args:
        in_refs: Input array references
        out_refs: Output array references
        num_keys: Number of sort keys
        descending: Sort in descending order
    """
    dim0, dim1 = _compute_padded_shape(*in_refs[0].shape, k=NUM_SUBLANES)
    dim0 = min(dim0, NUM_LANES)
    transpose_shape = (dim1 // (NUM_LANES // dim0), NUM_LANES)
           
    @functools.partial(pl.run_scoped, transpose_refs=[pltpu.VMEM(transpose_shape, to_32bit_dtype(x.dtype)) for x in in_refs])
    def _(transpose_refs):
      operands = [ref[...] for ref in in_refs]
      outs = bitonic_sort_arrays(
        operands,
        num_keys=num_keys,
        descending=descending,
        stage_unroll=stage_unroll,
        tile_unroll=tile_unroll,
        transpose_refs=transpose_refs,
      )
      for out, out_ref in zip(outs, out_refs, strict=True):
        out_ref[...] = out.astype(out_ref.dtype)




@functools.partial(
    jit,
    static_argnames=("num_keys", "descending", "interpret", "tile_unroll", "stage_unroll"),
)
def bitonic_sort(
    operand: jax.Array | Sequence[jax.Array],
    num_keys: int = 1,
    descending: bool = False,
    interpret: bool = False,
    stage_unroll: int | None = None,
    tile_unroll: int | None = None,  
) -> tuple[jax.Array, ...]:
    """
    Sort arrays using bitonic sort in compressed transpose format.

    Optimized for sorting power-of-2 sized arrays on TPU. Works entirely in
    compressed transpose format for maximum efficiency. Similar to bitonic_topk
    but performs full sort.

    Supports arbitrary input shapes - padding is handled automatically to
    nearest power of 2.

    Args:
        operand: Input array(s) of shape [batch, sort_dim].
                Can be a single array or sequence of arrays.
                Any sort_dim is supported (will be padded automatically).
        num_keys: Number of arrays to use as sort keys.
        descending: If True, sort in descending order.
        interpret: If True, run in CPU interpret mode.

    Returns:
        Tuple of arrays (same length as input operands):
            - Each array has same shape as input, sorted along axis 1

    Example:
        >>> import jax.numpy as jnp
        >>> from tallax._src.bitonic_topk import bitonic_sort
        >>> x = jnp.array([[3, 1, 4, 2], [8, 5, 7, 6]], dtype=jnp.int32)
        >>> result = bitonic_sort(x, descending=False)
        >>> print(result[0])
        [[1 2 3 4]
         [5 6 7 8]]
    """
    operands, unpadded_shape = canonicalize_operand(operand)
    # For ascending sort, pad with 'max' so padding values sort to the end
    # For descending sort, pad with 'min' so padding values sort to the end
    operands = [pad(x, (NUM_SUBLANES, NUM_LANES),
      val='min' if descending else 'max') for x in operands]
    batch_size, sort_dim = operands[0].shape

    # Define output shapes
    output_shapes = [
        jax.ShapeDtypeStruct((batch_size, sort_dim), op.dtype)
        for op in operands
    ]

    # Run normally without CSE
    pallas_fn = pl.pallas_call(
        functools.partial(
            bitonic_sort_refs,
            num_keys=num_keys,
            descending=descending,
            stage_unroll=stage_unroll,
            tile_unroll=tile_unroll,
        ),
        out_shape=(output_shapes,),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=int(0.9 * 2**27)
        ),
        interpret=interpret,
    )
    outputs = pallas_fn(operands)[0]

    return tuple(x[:unpadded_shape[0], :unpadded_shape[1]] for x in outputs)
