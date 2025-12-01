import functools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import lax
from jax.extend.random import threefry2x32_p

def _bits_to_uniform(bits, dtype):
    """
    Convert random uint32 bits to uniform float in [0, 1).

    This matches the conversion in jax._src.random._uniform().

    Args:
        bits: uint32 array of random bits
        dtype: Target float dtype

    Returns:
        Array of uniform random floats in [0, 1)
    """
    # Get dtype properties
    finfo = jnp.finfo(dtype)
    nbits = finfo.bits
    nmant = finfo.nmant

    # Right-shift to keep only mantissa bits
    # For float32: keep 23 bits, shift right by (32 - 23) = 9
    float_bits = jax.lax.shift_right_logical(
        bits,
        jnp.uint32(nbits - nmant)
    )

    # Create bit pattern for 1.0 in the target dtype
    # For float32: 0x3F800000 (sign=0, exp=127, mantissa=0)
    one_bits = jnp.ones((), dtype=dtype).view(jnp.uint32)

    # OR with 1.0 bit pattern to set exponent
    float_bits = jax.lax.bitwise_or(float_bits, one_bits)

    # Bitcast to float and subtract 1.0 to get [0, 1)
    floats = jax.lax.bitcast_convert_type(float_bits, dtype)
    return floats - jnp.ones((), dtype=dtype)

def sparse_random_uniform(
    key_ref: jax.Array,
    indices: Sequence[jax.Array],
    dim1_size: int,
    dtype=jnp.float32,
    minval=0.,
    maxval=1.
):
  """Generates uniform random numbers at specified sparse indices.

  Args:
    key_ref: Random key array. Must be 2D, typically shape (1, 2) inside Pallas
             kernels or when using refs.
    indices: Sequence of 2 arrays (row_indices, col_indices).
    dim1_size: Size of the second dimension (columns) for linear index calculation.
    dtype: Output dtype.
    minval: Minimum value.
    maxval: Maximum value.

  Returns:
    Array of random uniform values with shape broadcasted from indices.
  """
  if len(indices) != 2:
    raise ValueError(f"indices must be length 2, got {len(indices)}")

  if key_ref.ndim != 2:
    raise ValueError(f"key_ref must be 2D, got shape {key_ref.shape}")

  counts_lo = indices[0] * dim1_size + indices[1]
  counts_lo = counts_lo.astype(jnp.uint32)
  counts_hi = jnp.zeros_like(counts_lo)

  k1 = key_ref[0, 0]
  k2 = key_ref[0, 1]

  bits1, bits2 = threefry2x32_p.bind(
      k1, k2, counts_hi, counts_lo)
  bits = bits1 ^ bits2
  floats = _bits_to_uniform(bits, dtype)

  # Scale to [minval, maxval) following JAX's implementation
  minval = jax.lax.convert_element_type(minval, dtype)
  maxval = jax.lax.convert_element_type(maxval, dtype)

  # Scale and shift: floats * (maxval - minval) + minval
  # Use lax.max to ensure values are at least minval
  return jax.lax.max(minval, floats * (maxval - minval) + minval)

def sparse_random_categorical(
    key_ref: jax.Array,
    logits: jax.Array,
    indices: Sequence[jax.Array],
    dim1_size: int,
    axis: int = -1,
    dtype=jnp.float32
):
  """Sample from categorical distribution using Gumbel-max trick.

  Args:
    key_ref: Random key array (2D).
    logits: Logits array.
    indices: Indices for random generation.
    dim1_size: Size of dim1 for index linearization.
    axis: Axis to argmax over.
    dtype: Computation dtype.

  Returns:
    Argmax index after adding Gumbel noise.
  """
  if dtype != jnp.float32:
    raise NotImplementedError("Only float32 supported for now")

  u = sparse_random_uniform(key_ref, indices, dim1_size, dtype=dtype,
    minval=jnp.finfo(dtype).tiny)
  gumbel = -jnp.log(-jnp.log(u))
  return jnp.argmax(logits + gumbel, axis=axis)
