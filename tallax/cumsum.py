
import functools
import math
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .utils import (
    iota_tile,
    NUM_LANES,
    NUM_SUBLANES,
    canonicalize_operand
)

def _cumsum_kernel(x_ref, o_ref, *, m):
  x = x_ref[...]

  # We assume shape (8, 128). NUM_LANES=128.

  # Effective m for block size. Treat m=0 as 1.
  eff_m = m if m > 0 else 1

  # Phase 1: Intra-block Hillis-Steele (Short distance / Permutes)
  # Run steps 0 to log2(eff_m)-1
  # Use permutations (take_along_axis)
  # If eff_m=1, loop is empty.
  num_phase1_steps = int(math.log2(eff_m))

  for step in range(num_phase1_steps):
    offset = 1 << step
    idx = iota_tile(1)

    # Permutation logic: "permutes for short ones"
    # Calculate source indices.
    # We use modulo to ensure valid indices for take_along_axis.
    src_idx = (idx - offset) % NUM_LANES
    shifted_x = jnp.take_along_axis(x, src_idx, axis=1)

    # Mask logic
    # We only add if src is within the same block of size eff_m
    # block_start = (idx // eff_m) * eff_m
    # Condition: idx >= offset AND (idx - offset) >= block_start
    block_start = (idx // eff_m) * eff_m

    # Note: (idx - offset) check implicitly handles the idx >= offset case
    # if block_start is correctly computed for negative values, but iota is unsigned effectively here.
    # idx is int32.
    valid_src_mask = (idx >= offset) & ((idx - offset) >= block_start)

    x = jnp.where(valid_src_mask, x + shifted_x, x)

  # Phase 2: Block Merges (Long distance / Mask Sums)
  # "mask iota sums for long distance adds"
  # Start merging blocks of size eff_m, then 2*eff_m, etc.
  curr_size = eff_m
  while curr_size < NUM_LANES:
      # We merge pairs of blocks: (0,1), (2,3), ...
      # Block size becomes 2 * curr_size after merge.
      # Merge logic: Add sum of left block to all elements of right block.
      # Since left block is already locally cumsummed (either by Phase 1 or previous Phase 2 merges),
      # the last element of the left block contains the sum of that block.

      num_pairs = NUM_LANES // (curr_size * 2)
      for i in range(num_pairs):
          # Indices for the pair i
          left_block_end = (i * 2 + 1) * curr_size - 1
          right_block_start = (i * 2 + 1) * curr_size
          right_block_end = (i * 2 + 2) * curr_size

          # Extract adder using mask sum
          # "mixture of masks... mask iota sums"
          # Use mask to pick the value at left_block_end
          adder_mask = iota_tile(1) == left_block_end
          # Sum reduces along axis 1 (lanes), broadcasting to (8, 1).
          adder = (x * adder_mask).sum(axis=1, keepdims=True)

          # Add to right block
          # "between 64 and 128 or between 32 and 64" logic matches here
          dst_mask = (iota_tile(1) >= right_block_start) & (iota_tile(1) < right_block_end)
          x = jnp.where(dst_mask, x + adder, x)

      curr_size *= 2

  o_ref[...] = x


@functools.partial(
    jax.jit,
    static_argnames=('m', 'interpret')
)
def lax_cumsum_pallas(
    operand: jax.Array,
    m: int = 64,
    interpret: bool = False,
) -> jax.Array:
  """Compute cumulative sum along last dimension using Pallas.

  Args:
    operand: Input array of shape (8, 128).
    m: Threshold for switching between permutation and mask-based shifting.
       Offsets < m use permutations. Offsets >= m use masks.
       Must be a power of 2.
    interpret: Whether to run in interpreter mode.

  Returns:
    Array of same shape/dtype with cumulative sum.
  """
  operands, shape = canonicalize_operand(operand)

  # We expect exactly (8, 128)
  if shape != (NUM_SUBLANES, NUM_LANES):
      raise ValueError(f"Input shape must be ({NUM_SUBLANES}, {NUM_LANES}), got {shape}")

  # Check m is power of 2 (or 0)
  if m != 0 and (m & (m - 1) != 0):
      raise ValueError(f"m must be a power of 2, got {m}")

  operand = operands[0]

  grid = (1, 1)
  block_shape = (NUM_SUBLANES, NUM_LANES)

  in_specs = [pl.BlockSpec(block_shape, lambda i, j: (0, 0))]
  out_specs = pl.BlockSpec(block_shape, lambda i, j: (0, 0))

  return pl.pallas_call(
      functools.partial(_cumsum_kernel, m=m),
      out_shape=jax.ShapeDtypeStruct(shape, operand.dtype),
      in_specs=in_specs,
      out_specs=out_specs,
      grid=grid,
      interpret=interpret
  )(operand)
