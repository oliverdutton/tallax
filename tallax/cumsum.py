
import functools
import math
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax.utils import (
    iota_tile,
    NUM_LANES,
    NUM_SUBLANES,
    canonicalize_operand
)

def _cumsum_kernel(x_ref, o_ref, *, m):
  x = x_ref[...]

  # Effective m. m=0 treated as 1.
  eff_m = m if m > 0 else 1

  # Phase 1: Short distance / Permutes
  # Intra-block Hillis-Steele using take_along_axis
  for step in range(int(math.log2(eff_m))):
    offset = 1 << step
    idx = iota_tile(1)

    src_idx = (idx - offset) % NUM_LANES
    shifted_x = jnp.take_along_axis(x, src_idx, axis=1)

    # Only add if src is within the same eff_m block
    block_start = (idx // eff_m) * eff_m
    valid_src_mask = (idx >= offset) & ((idx - offset) >= block_start)
    x = jnp.where(valid_src_mask, x + shifted_x, x)

  # Phase 2: Long distance / Mask Sums
  # Merge blocks of size eff_m, 2*eff_m, etc.
  curr_size = eff_m
  while curr_size < NUM_LANES:
      # Merge pairs: (0,1), (2,3)... adding left block sum to right block
      num_pairs = NUM_LANES // (curr_size * 2)
      for i in range(num_pairs):
          base = (i * 2 + 1) * curr_size

          # Extract sum of left block (at base-1)
          adder = (x * (iota_tile(1) == (base - 1))).sum(axis=1, keepdims=True)

          # Add to right block [base, base+curr_size)
          dst_mask = (iota_tile(1) >= base) & (iota_tile(1) < (base + curr_size))
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

  if shape != (NUM_SUBLANES, NUM_LANES):
      raise ValueError(f"Input shape must be ({NUM_SUBLANES}, {NUM_LANES}), got {shape}")

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
