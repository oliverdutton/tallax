"""High-precision unsigned integer arithmetic using i32 arrays.

Supports arbitrary bit widths by splitting into multiple parts.
Tracks maximum value bounds to minimize normalization overhead.
"""

from dataclasses import dataclass
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
