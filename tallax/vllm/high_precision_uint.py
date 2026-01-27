"""High-precision 48-bit unsigned integer arithmetic using i32 arrays.

Specialized for 24-bit parts (2 parts = 48 bits total).
Tracks maximum value bounds to minimize harmonization.
"""

from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import tree_util


@dataclass
class U48:
  """Unsigned 48-bit integer with two 24-bit parts and dynamic bound tracking.

  Value = parts[0] + parts[1] * 2^24

  Attributes:
    parts: List of 2 i32 arrays [low_24bits, high_24bits]
    max_value_bound_per_part: List of upper bounds for each part (for overflow tracking)
  """
  parts: list[jax.Array]
  max_value_bound_per_part: list[int]

  @classmethod
  def from_i32_array(cls, x: jax.Array, max_val: int) -> 'U48':
    """Create from i32 array with values in [0, 2^31).

    Args:
      x: i32 array with non-negative values
      max_val: Maximum value in the array

    Returns:
      U48 with values split into two 24-bit parts
    """
    if max_val >= 2**48:
      raise ValueError(f"max_val must be < 2^48, got {max_val}")
    mask = 0xFFFFFF  # 24 bits
    low = x & mask
    high = (x >> 24) & mask
    # Track bounds independently: low is at most mask, high is at most max_val >> 24
    low_bound = mask
    high_bound = max_val >> 24
    return cls([low, high], max_value_bound_per_part=[low_bound, high_bound])

  @classmethod
  def from_f32(cls, x: jax.Array, max_val: int) -> 'U48':
    """Create from f32 by extracting 24-bit parts.

    Args:
      x: f32 array with non-negative values
      max_val: Maximum expected value

    Returns:
      U48 with extracted parts
    """
    if max_val >= 2**48:
      raise ValueError(f"max_val must be < 2^48, got {max_val}")
    modulo = jnp.float32(2**24)
    low = jnp.fmod(x, modulo).astype(jnp.int32)
    high = jnp.floor(x / modulo).astype(jnp.int32)
    # Track bounds independently: low is at most mask, high is at most max_val >> 24
    low_bound = 0xFFFFFF
    high_bound = max_val >> 24
    return cls([low, high], max_value_bound_per_part=[low_bound, high_bound])

  def to_f32(self) -> jax.Array:
    """Convert to f32."""
    return self.parts[0].astype(jnp.float32) + self.parts[1].astype(jnp.float32) * jnp.float32(2**24)

  def needs_harmonize(self) -> bool:
    """Check if harmonization is needed to prevent i32 overflow in any part."""
    return any(bound >= 2**31 for bound in self.max_value_bound_per_part)

  def harmonize(self) -> 'U48':
    """Propagate carries from low to high part, normalizing to 24-bit parts."""
    mask = 0xFFFFFF
    low_with_carry = self.parts[0]
    low_normalized = low_with_carry & mask
    carry = low_with_carry >> 24

    high_with_carry = self.parts[1] + carry
    high_normalized = high_with_carry & mask

    # Only 2 parts allowed (compile-time check on bound)
    # Maximum possible carry from low part is (max_value_bound_per_part[0] >> 24)
    # Maximum high_with_carry = max_value_bound_per_part[1] + (max_value_bound_per_part[0] >> 24)
    max_carry = self.max_value_bound_per_part[0] >> 24
    max_high_with_carry = self.max_value_bound_per_part[1] + max_carry
    if max_high_with_carry > mask:
      raise ValueError(
        f"Harmonization would require more than 2 parts: "
        f"max_value_bound_per_part={self.max_value_bound_per_part}, "
        f"max_carry={max_carry}, max_high_with_carry={max_high_with_carry} > {mask}"
      )
    
    # After harmonization, both parts are bounded by mask
    return U48([low_normalized, high_normalized], max_value_bound_per_part=[mask, mask])

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
    new_bounds = [bound * num_vals for bound in self.max_value_bound_per_part]

    result = U48([low_sum, high_sum], max_value_bound_per_part=new_bounds)
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
    new_bounds = [
      self_to_add.max_value_bound_per_part[i] + other_to_add.max_value_bound_per_part[i]
      for i in range(len(result_parts))
    ]
    result = U48(result_parts, max_value_bound_per_part=new_bounds)
    return result.harmonize() if result.needs_harmonize() else result

  def __lt__(self, other: 'U48') -> jax.Array:
    """Compare self < other."""
    assert len(self.parts) == len(other.parts), \
      f"Cannot compare U48 with different number of parts: {len(self.parts)} != {len(other.parts)}"
    assert len(self.parts) == 2, "Simplified __lt__ only supports 2 parts"

    # For 2-part comparison: self < other iff 
    # self[1] < other[1] OR (self[1] == other[1] AND self[0] < other[0])
    # Using bitwise ops: pi[1] < pj[1] | (pi[1] == pj[1] & pi[0] < pj[0])
    return (self.parts[1] < other.parts[1]) | ((self.parts[1] == other.parts[1]) & (self.parts[0] < other.parts[0]))


# Register U48 as a JAX PyTree for proper tracing in JIT-compiled functions
def _u48_tree_flatten(u48):
  """Flatten U48 into (children, aux_data) for JAX PyTree."""
  # children: dynamic values that JAX needs to trace (the array parts)
  # aux_data: static metadata (the integer bounds)
  return (u48.parts, u48.max_value_bound_per_part)


def _u48_tree_unflatten(aux_data, children):
  """Reconstruct U48 from flattened representation."""
  return U48(parts=children, max_value_bound_per_part=aux_data)


# Register the PyTree
tree_util.register_pytree_node(U48, _u48_tree_flatten, _u48_tree_unflatten)
