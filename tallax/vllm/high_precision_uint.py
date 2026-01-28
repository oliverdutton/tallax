"""High-precision unsigned integer arithmetic using i32 arrays.

Supports arbitrary bit widths by splitting into multiple parts.
Tracks maximum value bounds to minimize normalization overhead.

Also includes:
- u128 % u64 modulo using pure u32 arithmetic
- Random integer generation in [0, range) using modulo trick
"""

from dataclasses import dataclass
from typing import Tuple
import jax
import jax.numpy as jnp
from jax import tree_util


@dataclass(init=False)
class HighPrecisionUInt:
  """Unsigned integer with arbitrary bit width using multiple i32 parts.

  Value = sum(parts[i] * 2^(i * num_bits_per_part) for i in range(num_parts))

  Attributes:
    parts: List of i32 arrays representing the value in base 2^num_bits_per_part
    max_value_bound_per_part: List of upper bounds for each part
    num_bits_per_part: Number of bits each part represents when normalized (default: 24)
    total_bits: Total number of bits this integer can represent
  """

  parts: list[jax.Array]
  max_value_bound_per_part: list[int]
  num_bits_per_part: int
  total_bits: int

  def __init__(self, x_or_parts, max_val_or_bounds, num_bits_per_part: int = 24, total_bits: int | None = None):
    """Initialize HighPrecisionUInt.

    Args:
      x_or_parts: Either a jax.Array to split into parts, or list of existing parts
      max_val_or_bounds: Either a single max value (int), or list of per-part bounds
      num_bits_per_part: Number of bits per part when normalized (default: 24)
      total_bits: Total bit width (inferred if not provided)
    """
    self.num_bits_per_part = num_bits_per_part

    if isinstance(x_or_parts, (list, tuple)):
      # Constructed from existing parts
      self.parts = list(x_or_parts)
      self.max_value_bound_per_part = list(max_val_or_bounds)
      if total_bits is None:
        # Infer from number of parts
        self.total_bits = len(self.parts) * num_bits_per_part
      else:
        self.total_bits = total_bits
    else:
      # Initialization from array and max value
      if total_bits is None:
        # Infer total_bits from max_val_or_bounds
        total_bits = max(1, max_val_or_bounds.bit_length()) if isinstance(max_val_or_bounds, int) else 64

      self.total_bits = total_bits
      num_parts = (total_bits + num_bits_per_part - 1) // num_bits_per_part

      mask = (1 << num_bits_per_part) - 1
      self.parts = []
      self.max_value_bound_per_part = []

      x = x_or_parts
      max_val = max_val_or_bounds

      for i in range(num_parts):
        self.parts.append(x & mask)
        # Bound for this part
        shift = i * num_bits_per_part
        if i < num_parts - 1:
          # Intermediate parts are bounded by mask
          self.max_value_bound_per_part.append(mask)
        else:
          # Last part bounded by remaining bits of max_val
          self.max_value_bound_per_part.append(int(max_val >> shift))
        x = x >> num_bits_per_part

  @classmethod
  def from_i32_array(cls, x: jax.Array, max_val: int, num_bits_per_part: int = 24,
                     total_bits: int | None = None) -> 'HighPrecisionUInt':
    """Create from i32 array with values in [0, 2^31).

    Args:
      x: i32 array with non-negative values
      max_val: Maximum value in the array
      num_bits_per_part: Number of bits per part (default: 24)
      total_bits: Total bit width (inferred if not provided)

    Returns:
      HighPrecisionUInt with values split into parts
    """
    if total_bits is None:
      total_bits = max(1, max_val.bit_length())

    num_parts = (total_bits + num_bits_per_part - 1) // num_bits_per_part
    mask = (1 << num_bits_per_part) - 1

    parts = []
    bounds = []

    for i in range(num_parts):
      shift = i * num_bits_per_part
      parts.append((x >> shift) & mask)
      if i < num_parts - 1:
        # Intermediate parts are bounded by mask
        bounds.append(mask)
      else:
        # Last part bounded by remaining bits of max_val
        bounds.append(int(max_val >> shift))

    return cls(parts, bounds, num_bits_per_part, total_bits)

  @classmethod
  def from_f32(cls, x: jax.Array, max_val: int, num_bits_per_part: int = 24,
               total_bits: int | None = None) -> 'HighPrecisionUInt':
    """Create from f32 by extracting parts.

    Args:
      x: f32 array with non-negative values
      max_val: Maximum expected value
      num_bits_per_part: Number of bits per part (default: 24)
      total_bits: Total bit width (inferred if not provided)

    Returns:
      HighPrecisionUInt with extracted parts
    """
    if total_bits is None:
      total_bits = max(1, max_val.bit_length())

    num_parts = (total_bits + num_bits_per_part - 1) // num_bits_per_part
    mask = (1 << num_bits_per_part) - 1
    modulo = jnp.float32(1 << num_bits_per_part)

    parts = []
    bounds = []

    x_float = x
    for i in range(num_parts):
      part = jnp.fmod(x_float, modulo).astype(jnp.int32)
      parts.append(part)

      shift = i * num_bits_per_part
      if i < num_parts - 1:
        bounds.append(mask)
      else:
        bounds.append(int(max_val >> shift))

      x_float = jnp.floor(x_float / modulo)

    return cls(parts, bounds, num_bits_per_part, total_bits)

  def to_f32(self) -> jax.Array:
    """Convert to f32."""
    result = jnp.float32(0)
    multiplier = jnp.float32(1)
    base = jnp.float32(1 << self.num_bits_per_part)

    for part in self.parts:
      result = result + part.astype(jnp.float32) * multiplier
      multiplier = multiplier * base

    return result

  def to_u32_pair(self) -> Tuple[jax.Array, jax.Array]:
    """Convert to a pair of u32 values (high, low) representing a 64-bit value.

    Requires normalization first if needed. The value must fit in 64 bits.

    Returns:
      Tuple of (high_u32, low_u32) where value = high * 2^32 + low
    """
    normalized = self.normalize() if self.needs_normalize() else self

    # Reconstruct as 64-bit value from parts
    # Each part contributes num_bits_per_part bits
    low = jnp.uint32(0)
    high = jnp.uint32(0)

    for i, part in enumerate(normalized.parts):
      shift = i * normalized.num_bits_per_part
      part_u32 = part.astype(jnp.uint32)

      if shift < 32:
        # Part contributes to low word
        low = low | (part_u32 << shift)
        # Check if part spans into high word
        if shift + normalized.num_bits_per_part > 32:
          overflow_bits = shift + normalized.num_bits_per_part - 32
          high = high | (part_u32 >> (normalized.num_bits_per_part - overflow_bits))
      else:
        # Part contributes to high word
        high = high | (part_u32 << (shift - 32))

    return high, low

  @classmethod
  def from_u32_pair(
    cls,
    high: jax.Array,
    low: jax.Array,
    max_val: int = 2**48,
    num_bits_per_part: int = 32
  ) -> 'HighPrecisionUInt':
    """Create from a pair of u32 values representing a 64-bit value.

    Args:
      high: High 32 bits as u32 array
      low: Low 32 bits as u32 array
      max_val: Maximum expected value (default: 2^48)
      num_bits_per_part: Bits per part (default: 32 for u32 parts)

    Returns:
      HighPrecisionUInt representing high * 2^32 + low
    """
    if num_bits_per_part == 32:
      # Direct construction with 2 parts
      # Use conservative bounds that fit in int32 tracking
      parts = [low.astype(jnp.int32), high.astype(jnp.int32)]
      high_bound = min(2**31 - 1, int(max_val >> 32))
      bounds = [2**31 - 1, high_bound]
      return cls(parts, bounds, num_bits_per_part=32, total_bits=64)
    else:
      # Convert to desired num_bits_per_part
      combined = cls([low.astype(jnp.int32), high.astype(jnp.int32)],
                     [2**31 - 1, min(2**31 - 1, int(max_val >> 32))],
                     num_bits_per_part=32, total_bits=64)
      return combined

  def to_u32_quad(self) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Convert to four u32 values representing a 128-bit value.

    Returns:
      Tuple of (word0, word1, word2, word3) where:
        word0 = bits 127-96 (most significant)
        word1 = bits 95-64
        word2 = bits 63-32
        word3 = bits 31-0 (least significant)
    """
    normalized = self.normalize() if self.needs_normalize() else self

    # Initialize four 32-bit words
    words = [jnp.uint32(0), jnp.uint32(0), jnp.uint32(0), jnp.uint32(0)]

    for i, part in enumerate(normalized.parts):
      shift = i * normalized.num_bits_per_part
      part_u32 = part.astype(jnp.uint32)

      # Determine which word(s) this part contributes to
      # word3 = bits 0-31, word2 = bits 32-63, word1 = bits 64-95, word0 = bits 96-127
      word_idx = 3 - (shift // 32)
      bit_in_word = shift % 32

      if word_idx >= 0:
        words[word_idx] = words[word_idx] | (part_u32 << bit_in_word)

        # Check if part spans into next word
        if bit_in_word + normalized.num_bits_per_part > 32 and word_idx > 0:
          overflow_bits = bit_in_word + normalized.num_bits_per_part - 32
          words[word_idx - 1] = words[word_idx - 1] | (part_u32 >> (normalized.num_bits_per_part - overflow_bits))

    return tuple(words)

  @classmethod
  def from_u32_quad(
    cls,
    word0: jax.Array,
    word1: jax.Array,
    word2: jax.Array,
    word3: jax.Array,
    max_val: int = 2**96,
    num_bits_per_part: int = 32
  ) -> 'HighPrecisionUInt':
    """Create from four u32 values representing a 128-bit value.

    Args:
      word0: Bits 127-96 (most significant)
      word1: Bits 95-64
      word2: Bits 63-32
      word3: Bits 31-0 (least significant)
      max_val: Maximum expected value
      num_bits_per_part: Bits per part (default: 32)

    Returns:
      HighPrecisionUInt representing the 128-bit value
    """
    # Parts are ordered from LSB to MSB
    parts = [
      word3.astype(jnp.int32),
      word2.astype(jnp.int32),
      word1.astype(jnp.int32),
      word0.astype(jnp.int32),
    ]
    # Use conservative bounds that fit in int32 tracking
    high_bound = min(2**31 - 1, int(max_val >> 96))
    bounds = [2**31 - 1, 2**31 - 1, 2**31 - 1, high_bound]
    return cls(parts, bounds, num_bits_per_part=32, total_bits=128)

  def needs_normalize(self) -> bool:
    """Check if normalization is needed for correctness or overflow prevention."""
    mask = (1 << self.num_bits_per_part) - 1

    # Must normalize if any part (except the last) exceeds its bit allocation
    for i in range(len(self.parts) - 1):
      if self.max_value_bound_per_part[i] > mask:
        return True

    # Or if any part is approaching int32 limit (2^31)
    return any(bound >= 2**31 for bound in self.max_value_bound_per_part)

  # Alias for backward compatibility with U48
  def needs_harmonize(self) -> bool:
    """Alias for needs_normalize() for backward compatibility."""
    return self.needs_normalize()

  def normalize(self) -> 'HighPrecisionUInt':
    """Propagate carries from lower to higher parts, normalizing to num_bits_per_part bits."""
    mask = (1 << self.num_bits_per_part) - 1
    num_parts = len(self.parts)

    normalized_parts = []
    new_bounds = []
    carry = 0
    max_carry = 0

    for i in range(num_parts):
      part_with_carry = self.parts[i] + carry

      if i < num_parts - 1:
        # Intermediate parts: mask and propagate carry
        normalized_parts.append(part_with_carry & mask)
        carry = part_with_carry >> self.num_bits_per_part

        # Track maximum possible carry
        max_part_with_carry = self.max_value_bound_per_part[i] + max_carry
        new_bounds.append(mask)
        max_carry = max_part_with_carry >> self.num_bits_per_part
      else:
        # Last part: can grow beyond num_bits_per_part, but not beyond int32
        normalized_parts.append(part_with_carry)
        max_part_with_carry = self.max_value_bound_per_part[i] + max_carry

        if max_part_with_carry >= 2**31:
          raise ValueError(
            f"Normalization would overflow int32 in final part: "
            f"max_value_bound_per_part={self.max_value_bound_per_part}, "
            f"max_final_with_carry={max_part_with_carry} >= 2^31. "
            f"Consider using more parts or normalizing more frequently."
          )

        new_bounds.append(int(max_part_with_carry))

    return HighPrecisionUInt(normalized_parts, new_bounds, self.num_bits_per_part, self.total_bits)

  # Alias for backward compatibility with U48
  def harmonize(self) -> 'HighPrecisionUInt':
    """Alias for normalize() for backward compatibility."""
    return self.normalize()

  def sum(self, axis: int = 1, keepdims: bool = True) -> 'HighPrecisionUInt':
    """Sum along specified axis.

    Note: The caller must ensure that the summation does not overflow
    signed int32 before normalization can be applied.

    Args:
      axis: Axis to sum along (default: 1)
      keepdims: Whether to keep dimensions (default: True)

    Returns:
      New HighPrecisionUInt with summed values
    """
    summed_parts = [part.sum(axis=axis, keepdims=keepdims) for part in self.parts]

    num_vals = self.parts[0].shape[axis]
    new_bounds = [bound * num_vals for bound in self.max_value_bound_per_part]

    result = HighPrecisionUInt(summed_parts, new_bounds, self.num_bits_per_part, self.total_bits)
    return result.normalize() if result.needs_normalize() else result

  def __add__(self, other: 'HighPrecisionUInt') -> 'HighPrecisionUInt':
    """Add two HighPrecisionUInt, tracking per-part bounds and auto-normalizing when needed."""
    # Normalize if needed before adding
    self_to_add = self.normalize() if self.needs_normalize() else self
    other_to_add = other.normalize() if other.needs_normalize() else other

    assert len(self_to_add.parts) == len(other_to_add.parts), \
      f"Cannot add HighPrecisionUInt with different number of parts: {len(self_to_add.parts)} != {len(other_to_add.parts)}"
    assert self_to_add.num_bits_per_part == other_to_add.num_bits_per_part, \
      f"Cannot add HighPrecisionUInt with different num_bits_per_part: {self_to_add.num_bits_per_part} != {other_to_add.num_bits_per_part}"

    result_parts = [
      self_to_add.parts[i] + other_to_add.parts[i]
      for i in range(len(self_to_add.parts))
    ]

    new_bounds = [
      self_to_add.max_value_bound_per_part[i] + other_to_add.max_value_bound_per_part[i]
      for i in range(len(result_parts))
    ]

    result = HighPrecisionUInt(result_parts, new_bounds, self_to_add.num_bits_per_part, self_to_add.total_bits)
    return result.normalize() if result.needs_normalize() else result

  def __radd__(self, other):
    if other == 0:
      return self
    return self + other

  def __sub__(self, other: 'HighPrecisionUInt') -> 'HighPrecisionUInt':
    """Subtract two HighPrecisionUInt. Assumes self >= other."""
    s1 = self.normalize() if self.needs_normalize() else self
    s2 = other.normalize() if other.needs_normalize() else other

    assert len(s1.parts) == len(s2.parts), \
      f"Cannot subtract HighPrecisionUInt with different number of parts"
    assert s1.num_bits_per_part == s2.num_bits_per_part, \
      f"Cannot subtract HighPrecisionUInt with different num_bits_per_part"

    result_parts = []
    borrow = jnp.int32(0)

    for i in range(len(s1.parts)):
      part_diff = s1.parts[i] - s2.parts[i] - borrow

      if i < len(s1.parts) - 1:
        # Need to handle borrow for non-final parts
        borrow = (part_diff < 0).astype(jnp.int32)
        part_diff = part_diff + (borrow << s1.num_bits_per_part)

      result_parts.append(part_diff)

    return HighPrecisionUInt(
      result_parts,
      s1.max_value_bound_per_part,
      s1.num_bits_per_part,
      s1.total_bits
    )

  def __mul__(self, other: jax.Array) -> 'HighPrecisionUInt':
    """Multiply by a mask or scalar (array).

    Assumes 'other' is small (e.g., binary mask) to avoid overflow.
    """
    result_parts = [p * other for p in self.parts]
    return HighPrecisionUInt(
      result_parts,
      self.max_value_bound_per_part,
      self.num_bits_per_part,
      self.total_bits
    )

  def __rmul__(self, other: jax.Array) -> 'HighPrecisionUInt':
    return self.__mul__(other)

  def __lt__(self, other: 'HighPrecisionUInt') -> jax.Array:
    """Compare self < other. Normalizes both operands first for correctness."""
    s1 = self.normalize() if self.needs_normalize() else self
    s2 = other.normalize() if other.needs_normalize() else other

    assert len(s1.parts) == len(s2.parts), \
      f"Cannot compare HighPrecisionUInt with different number of parts"
    assert s1.num_bits_per_part == s2.num_bits_per_part, \
      f"Cannot compare HighPrecisionUInt with different num_bits_per_part"

    # Start with the most significant part
    # Initialize result from highest part comparison
    i = len(s1.parts) - 1
    less_at_i = s1.parts[i] < s2.parts[i]
    equal_at_i = s1.parts[i] == s2.parts[i]

    # result tracks whether s1 < s2 considering parts from i to end
    result = less_at_i
    equal_so_far = equal_at_i

    # Process remaining parts from high to low
    for i in range(len(s1.parts) - 2, -1, -1):
      less_at_i = s1.parts[i] < s2.parts[i]
      equal_at_i = s1.parts[i] == s2.parts[i]

      # If all higher parts are equal and this part is less, then s1 < s2
      result = result | (equal_so_far & less_at_i)

      # Update equal_so_far for next (lower) part
      equal_so_far = equal_so_far & equal_at_i

    return result


# Backward compatibility alias
U48 = HighPrecisionUInt


def modulo_u128_u64(dividend_u32: jax.Array, divisor_u32: jax.Array) -> Tuple[jax.Array, jax.Array]:
  """Compute (128-bit dividend) % (64-bit divisor) using only 32-bit operations.

  Uses binary long division algorithm, processing bits from MSB to LSB.

  Args:
    dividend_u32: Array of 4 uint32 representing 128-bit dividend
                  [word0 (bits 127-96), word1 (bits 95-64),
                   word2 (bits 63-32), word3 (bits 31-0)]
    divisor_u32: Array of 2 uint32 representing 64-bit divisor
                 [high (bits 63-32), low (bits 31-0)]

  Returns:
    Tuple of (rem_high, rem_low) as uint32, representing the 64-bit remainder
  """
  bh = divisor_u32[0]  # divisor high 32 bits
  bl = divisor_u32[1]  # divisor low 32 bits

  # Remainder registers: initially zero
  init_state = (jnp.uint32(0), jnp.uint32(0))

  def body_fun(i, state):
    rh, rl = state
    bit_idx = 127 - i  # Process from MSB (bit 127) to LSB (bit 0)

    # Extract bit from the 128-bit dividend array
    # Bit 127-96 is in word 0, bit 95-64 is in word 1, etc.
    word_idx = 3 - (bit_idx // 32)
    bit_in_word = bit_idx % 32
    bit = (dividend_u32[word_idx] >> bit_in_word) & jnp.uint32(1)

    # Shift remainder left by 1 and insert the next dividend bit at LSB
    # After this operation, the remainder could temporarily be 65 bits
    new_rh = (rh << 1) | (rl >> 31)
    new_rl = (rl << 1) | bit

    # Detect overflow: check if bit 64 would be set after the shift
    # This happens when rh had its MSB (bit 31) set before shifting
    overflow = (rh >> 31) & jnp.uint32(1)

    # 64-bit Comparison: (overflow:new_rh:new_rl) >= (0:bh:bl)
    # Three cases for is_greater:
    # 1. overflow==1: We have a 65-bit number, definitely >= 64-bit divisor
    # 2. new_rh > bh: Upper 32 bits are greater
    # 3. new_rh == bh AND new_rl >= bl: Upper bits equal, lower bits greater/equal
    is_greater = (
      (overflow == jnp.uint32(1)) |
      (new_rh > bh) |
      ((new_rh == bh) & (new_rl >= bl))
    )

    # 64-bit Subtraction: (new_rh:new_rl) - (bh:bl)
    # Handle borrow from low word to high word
    borrow = jnp.where(new_rl < bl, jnp.uint32(1), jnp.uint32(0))
    sub_rh = new_rh - bh - borrow
    sub_rl = new_rl - bl

    # Update remainder: subtract if greater/equal, otherwise keep shifted value
    rh_next = jnp.where(is_greater, sub_rh, new_rh)
    rl_next = jnp.where(is_greater, sub_rl, new_rl)

    return (rh_next, rl_next)

  # Process all 128 bits
  final_rh, final_rl = jax.lax.fori_loop(0, 128, body_fun, init_state)

  return final_rh, final_rl


def random_int_in_range(
  rng_key: jax.Array,
  range_high: jax.Array,
  range_low: jax.Array,
  shape: Tuple[int, ...] = ()
) -> Tuple[jax.Array, jax.Array]:
  """Generate uniformly distributed random integers in [0, range).

  Uses the modulo trick: generate random u128 values and take modulo range.
  This produces nearly uniform distribution when range << 2^128.

  Args:
    rng_key: JAX random key
    range_high: High 32 bits of the range (exclusive upper bound)
    range_low: Low 32 bits of the range
    shape: Shape of the output arrays (each element is a random value)

  Returns:
    Tuple of (result_high, result_low) as uint32 arrays representing
    random values in [0, range_high:range_low)
  """
  # Generate 4 random u32 values to form a u128
  keys = jax.random.split(rng_key, 4)

  # Generate random bits for each 32-bit word
  word0 = jax.random.bits(keys[0], shape=shape, dtype=jnp.uint32)
  word1 = jax.random.bits(keys[1], shape=shape, dtype=jnp.uint32)
  word2 = jax.random.bits(keys[2], shape=shape, dtype=jnp.uint32)
  word3 = jax.random.bits(keys[3], shape=shape, dtype=jnp.uint32)

  # Stack into dividend array [word0, word1, word2, word3]
  dividend = jnp.stack([word0, word1, word2, word3], axis=-1)

  # Divisor is the range [range_high, range_low]
  divisor = jnp.stack([range_high, range_low], axis=-1)

  # Vectorized modulo operation
  def single_modulo(div_dis):
    div, dis = div_dis
    return modulo_u128_u64(div, dis)

  if shape == ():
    # Scalar case
    result_high, result_low = modulo_u128_u64(
      jnp.array([word0, word1, word2, word3]),
      jnp.array([range_high, range_low])
    )
  else:
    # Batched case - use vmap
    # Reshape divisor to broadcast
    divisor_broadcast = jnp.broadcast_to(
      jnp.stack([range_high, range_low]),
      shape + (2,)
    )

    # vmap over all dimensions
    def batched_modulo(dividend_flat, divisor_flat):
      return modulo_u128_u64(dividend_flat, divisor_flat)

    # Flatten batch dimensions for vmap
    flat_shape = (-1,)
    dividend_flat = dividend.reshape(flat_shape + (4,))
    divisor_flat = divisor_broadcast.reshape(flat_shape + (2,))

    # Apply vmap
    result_high_flat, result_low_flat = jax.vmap(batched_modulo)(
      dividend_flat, divisor_flat
    )

    # Reshape back
    result_high = result_high_flat.reshape(shape)
    result_low = result_low_flat.reshape(shape)

  return result_high, result_low


def random_high_precision_uint(
  rng_key: jax.Array,
  max_val: int,
  shape: Tuple[int, ...] = (),
  num_bits_per_part: int = 32
) -> HighPrecisionUInt:
  """Generate random HighPrecisionUInt values uniformly in [0, max_val).

  Args:
    rng_key: JAX random key
    max_val: Exclusive upper bound (must be <= 2^64)
    shape: Shape of the output (each element is a random HighPrecisionUInt)
    num_bits_per_part: Bits per part in the result

  Returns:
    HighPrecisionUInt with random values in [0, max_val)
  """
  if max_val > 2**64:
    raise ValueError(f"max_val must be <= 2^64, got {max_val}")

  range_high = jnp.uint32(max_val >> 32)
  range_low = jnp.uint32(max_val & 0xFFFFFFFF)

  result_high, result_low = random_int_in_range(
    rng_key, range_high, range_low, shape
  )

  return HighPrecisionUInt.from_u32_pair(
    result_high, result_low,
    max_val=max_val - 1,
    num_bits_per_part=num_bits_per_part
  )


# Register HighPrecisionUInt as a JAX PyTree for proper tracing in JIT-compiled functions
def _high_precision_uint_tree_flatten(hpu):
  """Flatten HighPrecisionUInt into (children, aux_data) for JAX PyTree."""
  # children: dynamic values that JAX needs to trace (the array parts)
  # aux_data: static metadata (bounds, num_bits_per_part, total_bits)
  return (hpu.parts, (hpu.max_value_bound_per_part, hpu.num_bits_per_part, hpu.total_bits))


def _high_precision_uint_tree_unflatten(aux_data, children):
  """Reconstruct HighPrecisionUInt from flattened representation."""
  max_value_bound_per_part, num_bits_per_part, total_bits = aux_data
  return HighPrecisionUInt(children, max_value_bound_per_part, num_bits_per_part, total_bits)


# Register the PyTree
tree_util.register_pytree_node(
  HighPrecisionUInt,
  _high_precision_uint_tree_flatten,
  _high_precision_uint_tree_unflatten
)
