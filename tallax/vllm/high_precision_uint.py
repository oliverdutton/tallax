"""High-precision 48-bit unsigned integer arithmetic using i32 arrays.

Specialized for 24-bit parts (2 parts = 48 bits total).
Tracks maximum value bounds to minimize harmonization.
"""

from dataclasses import dataclass
import jax
import jax.numpy as jnp


@dataclass
class U48:
  """Unsigned 48-bit integer with two 24-bit parts and dynamic bound tracking.

  Value = parts[0] + parts[1] * 2^24

  Attributes:
    parts: List of 2 i32 arrays [low_24bits, high_24bits]
    max_value_bound_per_part: Upper bound on each part (for overflow tracking)
  """
  parts: list[jax.Array]
  max_value_bound_per_part: int

  @classmethod
  def from_i32_array(cls, x: jax.Array, max_val: int) -> 'U48':
    """Create from i32 array with values in [0, 2^31).

    Args:
      x: i32 array with non-negative values
      max_val: Maximum value in the array

    Returns:
      U48 with values split into two 24-bit parts
    """
    mask = 0xFFFFFF  # 24 bits
    low = x & mask
    high = (x >> 24) & mask
    # Bound per part: low is at most mask, high is at most max_val >> 24
    bound_per_part = max(mask, max_val >> 24) if max_val > mask else mask
    return cls([low, high], max_value_bound_per_part=bound_per_part)

  @classmethod
  def from_f32(cls, x: jax.Array, max_val: int) -> 'U48':
    """Create from f32 by extracting 24-bit parts.

    Args:
      x: f32 array with non-negative values
      max_val: Maximum expected value

    Returns:
      U48 with extracted parts
    """
    modulo = jnp.float32(2**24)
    low = jnp.fmod(x, modulo).astype(jnp.int32)
    high = jnp.floor(x / modulo).astype(jnp.int32)
    bound_per_part = max(0xFFFFFF, max_val >> 24) if max_val > 0xFFFFFF else 0xFFFFFF
    return cls([low, high], max_value_bound_per_part=bound_per_part)

  def to_f32(self) -> jax.Array:
    """Convert to f32."""
    return self.parts[0].astype(jnp.float32) + self.parts[1].astype(jnp.float32) * jnp.float32(2**24)

  def needs_harmonize(self) -> bool:
    """Check if harmonization is needed to prevent i32 overflow in any part."""
    return self.max_value_bound_per_part >= 2**31

  def harmonize(self) -> 'U48':
    """Propagate carries from low to high part, normalizing to 24-bit parts."""
    mask = 0xFFFFFF
    low_with_carry = self.parts[0]
    low_normalized = low_with_carry & mask
    carry = low_with_carry >> 24

    high_with_carry = self.parts[1] + carry
    high_normalized = high_with_carry & mask
    overflow = high_with_carry >> 24

    return U48([low_normalized, high_normalized, overflow], max_value_bound_per_part=mask)

  def sum(self, axis: int = 1, keepdims: bool = True) -> 'U48':
    """Sum along axis 1 for (batch, NUM_LANES) shaped arrays.

    Args:
      axis: Must be 1
      keepdims: Whether to keep dimension

    Returns:
      U48 with summed values
    """
    assert axis == 1
    assert len(self.parts) == 2

    # Sum each part
    low_sum = self.parts[0].sum(axis=1, keepdims=keepdims)
    high_sum = self.parts[1].sum(axis=1, keepdims=keepdims)

    # Track new bound per part: each part's bound * NUM_LANES
    num_vals = self.parts[0].shape[1]
    new_bound_per_part = self.max_value_bound_per_part * num_vals

    result = U48([low_sum, high_sum], max_value_bound_per_part=new_bound_per_part)
    return result.harmonize() if result.needs_harmonize() else result

  def __add__(self, other: 'U48') -> 'U48':
    """Add two U48, tracking per-part bounds and auto-harmonizing when needed."""
    # Harmonize if needed before adding
    self_to_add = self.harmonize() if self.needs_harmonize() else self
    other_to_add = other.harmonize() if other.needs_harmonize() else other

    # Add corresponding parts
    assert len(self_to_add.parts) == len(other_to_add.parts), \
      f"Cannot add U48 with different number of parts: {len(self_to_add.parts)} != {len(other_to_add.parts)}"

    result_parts = [
      self_to_add.parts[i] + other_to_add.parts[i]
      for i in range(len(self_to_add.parts))
    ]

    # Track new bound per part: sum of per-part bounds
    new_bound_per_part = self_to_add.max_value_bound_per_part + other_to_add.max_value_bound_per_part
    result = U48(result_parts, max_value_bound_per_part=new_bound_per_part)
    return result.harmonize() if result.needs_harmonize() else result

  def __lt__(self, other: 'U48') -> jax.Array:
    """Compare self < other."""
    assert len(self.parts) == len(other.parts), \
      f"Cannot compare U48 with different number of parts: {len(self.parts)} != {len(other.parts)}"

    # Compare from MSB to LSB using parts[0] as template
    result = jnp.zeros_like(self.parts[0], dtype=bool)
    still_equal = jnp.ones_like(self.parts[0], dtype=bool)

    for i in range(len(self.parts) - 1, -1, -1):
      part_gt = self.parts[i] > other.parts[i]
      part_eq = self.parts[i] == other.parts[i]
      result |= still_equal & part_gt
      still_equal &= part_eq

    # self >= other, so invert for <
    return ~(result | still_equal)
