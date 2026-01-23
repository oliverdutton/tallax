"""High-precision i64 utilities for summation without overflow.

Provides utilities for performing i64-precision summations on u32 arrays by splitting
values into 16-bit parts and carefully tracking carries across reductions.
"""

import jax
import jax.numpy as jnp


def _chunk_and_stack(x: jax.Array, split_dim: int, chunk_size: int, stack_dim: int = 0, pad_val: int = 0) -> jax.Array:
  """Split array into chunks along a dimension, pad last chunk if needed, and stack."""
  dim_len = x.shape[split_dim]
  num_full_chunks = dim_len // chunk_size
  remainder = dim_len % chunk_size

  if remainder == 0:
    chunks = jnp.split(x, num_full_chunks, axis=split_dim)
  else:
    full_chunk_len = num_full_chunks * chunk_size
    if num_full_chunks > 0:
      slices = [slice(None)] * x.ndim
      slices[split_dim] = slice(None, full_chunk_len)
      full_chunks = jnp.split(x[tuple(slices)], num_full_chunks, axis=split_dim)
    else:
      full_chunks = []

    slices = [slice(None)] * x.ndim
    slices[split_dim] = slice(full_chunk_len, None)
    pad_width = [(0, 0)] * x.ndim
    pad_width[split_dim] = (0, chunk_size - remainder)
    remainder_padded = jnp.pad(x[tuple(slices)], pad_width, constant_values=pad_val)
    chunks = full_chunks + [remainder_padded]

  return jnp.stack(chunks, axis=stack_dim)


def _split_to_i16s(i32s: list[jax.Array]) -> list[jax.Array]:
  """Split i32 arrays into i16 parts (LSB first). Masks after shift to avoid sign extension."""
  i16s = []
  for i32_arr in i32s:
    i16s.append(i32_arr & 0xFFFF)
    i16s.append((i32_arr >> 16) & 0xFFFF)
  return i16s


def _harmonize_i16s(i16s: list[jax.Array]) -> list[jax.Array]:
  """Propagate carries through i16 parts from LSB to MSB."""
  if not i16s:
    return []

  result = []
  carry = jnp.zeros_like(i16s[0])
  for i16 in i16s:
    i16_with_carry = i16 + carry
    result.append(i16_with_carry & 0xFFFF)
    carry = i16_with_carry >> 16
  result.append(carry)
  return result


def _combine_i16s_to_i32s(i16s: list[jax.Array]) -> list[jax.Array]:
  """Combine i16 parts into i32s (pairs from LSB to MSB)."""
  i32s = []
  for i in range(0, len(i16s), 2):
    i32 = (i16s[i + 1] << 16) | i16s[i] if i + 1 < len(i16s) else i16s[i]
    i32s.append(i32)
  return i32s


def i64_sum_dim1(x: jax.Array, chunk_size: int = 128):
  """Sum u32 array along axis=1 with i64 precision using two-stage reduction.

  Splits u32 values into 16-bit parts, sums separately to avoid overflow, then harmonizes
  carries. Only supports non-negative integers (requires uint32 dtype).

  Algorithm:
    1. Chunk input along axis=1 into tiles of size chunk_size (pad last with zeros)
    2. Split each u32 into two i16 parts, sum over chunks dimension -> (n, chunk_size) i64s
    3. Split i64s into i16 parts, sum over chunk_size dimension -> (n, 1) i64

  The key insight: splitting into 16-bit parts allows summing up to 32k (2^15) values
  without overflow, since (2^16-1) * 2^15 < 2^31.

  Args:
    x: u32 array of shape (n, m) where m < 2^31
    chunk_size: Tile size (default 128, must be < 32k)

  Returns:
    List of i32 arrays representing i64 sum of shape (n, 1), LSB first.
    E.g., [low_i32, high_i32] where value = low + high * 2^32

  Constraints:
    - x.dtype must be uint32
    - x.ndim must be 2
    - chunk_size < 32768 (2^15)
    - num_chunks = ceil(m / chunk_size) < 32768 (2^15)
    - Result can hold sums up to (2^32-1) * 32k * chunk_size ≈ 2^57 (for chunk_size=128)

  Example:
    >>> x = jnp.arange(256, dtype=jnp.uint32).reshape(2, 128)
    >>> i32s = i64_sum_dim1(x)
    >>> # i32s[0][i, 0] + i32s[1][i, 0] * 2^32 equals sum of row i
  """
  if x.dtype != jnp.uint32:
    raise NotImplementedError("Only supports uint32. Cast your array to uint32 first.")
  assert x.ndim == 2 and x.shape[1] < 2**31
  assert chunk_size < 32768, "chunk_size must be < 32k (2^15)"
  num_chunks = (x.shape[1] + chunk_size - 1) // chunk_size
  assert num_chunks < 32768, f"num_chunks={num_chunks} must be < 32k (2^15)"

  x_stacked = _chunk_and_stack(x, split_dim=1, stack_dim=0, chunk_size=chunk_size, pad_val=0)
  i16s = _split_to_i16s([x_stacked])
  i16s = _harmonize_i16s([i16.sum(axis=0) for i16 in i16s])
  i32s = _combine_i16s_to_i32s(_harmonize_i16s([i16.sum(axis=1, keepdims=True) for i16 in i16s]))
  return i32s
