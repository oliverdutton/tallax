"""High-precision unsigned integer arithmetic using i32 arrays.

This module provides a safe way to work with high-precision unsigned integers
using JAX i32 arrays. The key constraint is:
- Each part is < 2^16 (not full 32 bits)
- Can safely sum up to 2^15 values before harmonizing carries

This allows us to use i32 dtype while simulating u32, u48, u64, u96, etc.
depending on how many parts we use.
"""

from dataclasses import dataclass
import jax
import jax.numpy as jnp


@dataclass
class HighPrecisionUInt:
  """High-precision unsigned integer using i32 parts < 2^16.

  The value is represented as: sum(parts[i] * 2^(16*i) for i in range(len(parts)))

  Each part is constrained to < 2^16 to allow safe summation of up to 2^15 values
  before needing to harmonize carries. This is because:
    (2^16 - 1) * 2^15 = 2^31 - 2^15 < 2^31 (fits in positive i32)

  Attributes:
    parts: List of i32 arrays from LSB to MSB, each part < 2^16
  """
  parts: list[jax.Array]

  @classmethod
  def from_i32_array(cls, x: jax.Array) -> 'HighPrecisionUInt':
    """Create from i32 array containing values in [0, 2^31).

    Args:
      x: i32 array with non-negative values

    Returns:
      HighPrecisionUInt with each i32 split into two 16-bit parts
    """
    # Split each i32 into two 16-bit parts
    low = x & 0xFFFF
    high = (x >> 16) & 0xFFFF
    return cls([low, high])

  def to_f32(self) -> jax.Array:
    """Convert to f32.

    Returns:
      f32 array with same shape as each part
    """
    result = self.parts[0].astype(jnp.float32)
    scale = jnp.float32(2**16)
    for part in self.parts[1:]:
      result += part.astype(jnp.float32) * scale
      scale *= jnp.float32(2**16)
    return result

  @classmethod
  def from_f32(cls, x: jax.Array, num_parts: int) -> 'HighPrecisionUInt':
    """Create from f32 by extracting 16-bit parts.

    Args:
      x: f32 array with non-negative values
      num_parts: Number of 16-bit parts to create

    Returns:
      HighPrecisionUInt with specified number of parts
    """
    parts = []
    remainder = x
    for _ in range(num_parts):
      part_f32 = jnp.fmod(remainder, jnp.float32(2**16))
      parts.append(part_f32.astype(jnp.int32))
      remainder = jnp.floor(remainder / jnp.float32(2**16))
    return cls(parts)

  def harmonize(self) -> 'HighPrecisionUInt':
    """Propagate carries from LSB to MSB, ensuring each part < 2^16.

    Returns:
      New HighPrecisionUInt with carries propagated
    """
    result = []
    carry = jnp.zeros_like(self.parts[0], dtype=jnp.int32)
    for part in self.parts:
      part_with_carry = part + carry
      result.append(part_with_carry & 0xFFFF)
      carry = part_with_carry >> 16
    # Add final carry if non-zero
    if len(self.parts) > 0:
      result.append(carry)
    return HighPrecisionUInt(result)

  def sum(self, axis: int = 1, chunk_size: int = 128) -> 'HighPrecisionUInt':
    """Sum along specified axis.

    Currently only supports axis=1 with chunking to prevent overflow.
    Splits the dimension into chunks of size chunk_size (default 128),
    sums each chunk separately, then harmonizes carries.

    Theory: Each part is < 2^16. We can safely sum up to 2^15 such values:
      (2^16 - 1) * 2^15 = 2^31 - 2^15 < 2^31 (fits in positive i32)

    Two-stage reduction:
      1. Split vocab into chunks, sum num_chunks values -> requires num_chunks <= 2^15
      2. After harmonize, sum chunk_size values -> requires chunk_size <= 2^15

    Args:
      axis: Axis to sum along (must be 1)
      chunk_size: Size of each chunk (default 128, must be <= 2^15)

    Returns:
      HighPrecisionUInt with reduced dimension
    """
    if len(self.parts) == 0:
      raise ValueError("Cannot sum empty HighPrecisionUInt")

    if axis != 1:
        raise NotImplementedError(
            f"HighPrecisionUInt.sum currently only supports axis=1, got axis={axis}"
        )

    vocab_size = self.parts[0].shape[1]
    num_chunks = (vocab_size + chunk_size - 1) // chunk_size

    # Check both constraints
    if chunk_size > 2**15:
      raise ValueError(
        f"chunk_size={chunk_size} exceeds 2^15 (32768). "
        f"In the second reduction stage, we sum {chunk_size} values (each < 2^16). "
        f"For safety: chunk_size * (2^16-1) < 2^31 requires chunk_size <= 2^15."
      )

    if num_chunks > 2**15:
      raise ValueError(
        f"num_chunks={num_chunks} exceeds 2^15 (32768). "
        f"With vocab_size={vocab_size} and chunk_size={chunk_size}, "
        f"we have {num_chunks} chunks. In the first reduction, we sum these chunks. "
        f"For safety: num_chunks * (2^16-1) < 2^31 requires num_chunks <= 2^15. "
        f"Solution: increase chunk_size or reduce vocab_size."
      )

    # Chunk along axis 1
    result_parts = []
    for part in self.parts:
      # Split into chunks and stack
      dim_len = part.shape[1]
      num_full_chunks = dim_len // chunk_size
      remainder = dim_len % chunk_size

      if remainder == 0:
        chunks = jnp.split(part, num_full_chunks, axis=1)
      else:
        full_chunk_len = num_full_chunks * chunk_size
        if num_full_chunks > 0:
          full_chunks = jnp.split(part[:, :full_chunk_len], num_full_chunks, axis=1)
        else:
          full_chunks = []

        # Pad remainder chunk
        remainder_chunk = part[:, full_chunk_len:]
        pad_width = [(0, 0), (0, chunk_size - remainder)]
        remainder_padded = jnp.pad(remainder_chunk, pad_width, constant_values=0)
        chunks = full_chunks + [remainder_padded]

      # Stack chunks
      part_stacked = jnp.stack(chunks, axis=0)

      # Sum over chunks dimension (axis=0)
      part_summed = part_stacked.sum(axis=0, keepdims=False)
      result_parts.append(part_summed)

    # Harmonize carries across parts
    result = HighPrecisionUInt(result_parts).harmonize()

    # Sum along remaining dimension (axis=1) and harmonize
    return HighPrecisionUInt([
      part.sum(axis=1, keepdims=True) for part in result.parts
    ]).harmonize()

  def compare_ge(self, other: 'HighPrecisionUInt') -> jax.Array:
    """Compare self >= other.

    Compares from MSB to LSB. All values are non-negative.

    Args:
      other: HighPrecisionUInt to compare against

    Returns:
      Boolean array where self >= other
    """
    # Pad to same length using scalar 0 (will broadcast)
    max_len = max(len(self.parts), len(other.parts))
    self_parts = self.parts + [0] * (max_len - len(self.parts))
    other_parts = other.parts + [0] * (max_len - len(other.parts))

    # Compare from MSB to LSB
    result = jnp.ones_like(self.parts[0], dtype=bool)
    still_equal = jnp.ones_like(self.parts[0], dtype=bool)

    for i in range(max_len - 1, -1, -1):
      part_greater = self_parts[i] > other_parts[i]
      part_less = self_parts[i] < other_parts[i]
      part_equal = self_parts[i] == other_parts[i]

      # Update result: if still equal and current part is greater, set true
      # if still equal and current part is less, set false
      result = jnp.where(still_equal & part_greater, True, result)
      result = jnp.where(still_equal & part_less, False, result)

      # Update still_equal
      still_equal = still_equal & part_equal

    return result
