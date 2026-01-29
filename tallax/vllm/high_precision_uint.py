"""High-precision 48-bit unsigned integer arithmetic using two i32 parts.

Specialized for 48-bit (24 bits per part) for efficiency and deterministic
behavior across platforms.
"""

from dataclasses import dataclass
from typing import Tuple, List, Union, Optional
import jax
import jax.numpy as jnp
from jax import tree_util


@dataclass(init=False)
class U48:
  """48-bit unsigned integer using two i32 parts (24 bits each).

  Value = parts[0] + parts[1] * 2^24

  Attributes:
    parts: List of 2 i32 arrays representing the value
    max_value_bound_per_part: List of 2 upper bounds for each part
  """

  parts: list[jax.Array]
  max_value_bound_per_part: list[int]

  # Hardcoded constants
  BITS_PER_PART = 24
  TOTAL_BITS = 48

  def __init__(self, x_or_parts, max_val=None, **kwargs):
    """Initialize U48.

    Args:
      x_or_parts: Either a jax.Array to split into parts, or list of existing parts
      max_val: Either a single max value (int), or list of per-part bounds
      **kwargs: Ignored (for backward compatibility)
    """
    if isinstance(x_or_parts, (list, tuple)):
      # Constructed from existing parts
      self.parts = list(x_or_parts)
      if isinstance(max_val, (list, tuple)):
        self.max_value_bound_per_part = list(max_val)
      else:
        # Fallback if only single max_val provided for parts
        mask = (1 << 24) - 1
        self.max_value_bound_per_part = [mask, int((max_val or (2**48-1)) >> 24)]
    else:
      # Initialization from array and max value
      if max_val is None:
        max_val = 2**48 - 1
      
      mask = (1 << 24) - 1
      self.parts = [x_or_parts & mask, x_or_parts >> 24]
      
      if isinstance(max_val, (list, tuple)):
        self.max_value_bound_per_part = list(max_val)
      else:
        self.max_value_bound_per_part = [mask, int(max_val >> 24)]

  @classmethod
  def from_i32_array(cls, x: jax.Array, max_val: int, **kwargs) -> 'U48':
    """Create from i32 array by splitting into 24-bit parts."""
    mask = (1 << 24) - 1
    parts = [x & mask, x >> 24]
    bounds = [mask, int(max_val >> 24)]
    return cls(parts, bounds)

  @classmethod
  def from_f32(cls, x: jax.Array, max_val: int, **kwargs) -> 'U48':
    """Create from f32 by extracting 24-bit parts."""
    mask = (1 << 24) - 1
    modulo = float(1 << 24)
    parts = [
      jnp.fmod(x, modulo).astype(jnp.int32),
      jnp.floor(x / modulo).astype(jnp.int32)
    ]
    bounds = [mask, int(max_val >> 24)]
    return cls(parts, bounds)

  def to_f32(self) -> jax.Array:
    """Convert back to f32."""
    base = float(1 << 24)
    return self.parts[0].astype(jnp.float32) + self.parts[1].astype(jnp.float32) * base

  def to_u32_pair(self) -> Tuple[jax.Array, jax.Array]:
    """Convert to a pair of u32 values (high, low) representing a 64-bit value."""
    normalized = self.normalize() if self.needs_normalize() else self
    # low 32 bits: bits 0-23 from parts[0], bits 24-31 from bits 0-7 of parts[1]
    low = normalized.parts[0].astype(jnp.uint32) | ((normalized.parts[1].astype(jnp.uint32) & 0xFF) << 24)
    # high 32 bits: bits 32-47 from bits 8-23 of parts[1]
    high = normalized.parts[1].astype(jnp.uint32) >> 8
    return high, low

  @classmethod
  def from_u32_pair(cls, high: jax.Array, low: jax.Array, max_val: int = 2**48, **kwargs) -> 'U48':
    """Create from a pair of u32 values."""
    mask24 = (1 << 24) - 1
    p0 = (low & mask24).astype(jnp.int32)
    p1 = ((low >> 24) | (high << 8)).astype(jnp.int32)
    return cls([p0, p1], [mask24, int(max_val >> 24)])

  def needs_normalize(self) -> bool:
    """Check if carry propagation is needed."""
    mask = (1 << 24) - 1
    if self.max_value_bound_per_part[0] > mask:
      return True
    return any(b >= 2**31 for b in self.max_value_bound_per_part)

  def normalize(self) -> 'U48':
    """Propagate carries from parts[0] to parts[1]."""
    mask = (1 << 24) - 1
    carry = self.parts[0] >> 24
    new_parts = [self.parts[0] & mask, self.parts[1] + carry]
    
    max_carry = self.max_value_bound_per_part[0] >> 24
    new_max_high = self.max_value_bound_per_part[1] + max_carry
    
    if new_max_high >= 2**31:
      # We allow it in some cases but let's be safe.
      pass
      
    return U48(new_parts, [mask, int(new_max_high)])

  def sum(self, axis: int = 1, keepdims: bool = True) -> 'U48':
    """Sum along specified axis."""
    summed_parts = [p.sum(axis=axis, keepdims=keepdims) for p in self.parts]
    num_vals = self.parts[0].shape[axis]
    new_bounds = [b * num_vals for b in self.max_value_bound_per_part]
    result = U48(summed_parts, new_bounds)
    return result.normalize() if result.needs_normalize() else result

  def __add__(self, other: 'U48') -> 'U48':
    s1 = self.normalize() if self.needs_normalize() else self
    s2 = other.normalize() if other.needs_normalize() else other
    parts = [s1.parts[0] + s2.parts[0], s1.parts[1] + s2.parts[1]]
    bounds = [s1.max_value_bound_per_part[0] + s2.max_value_bound_per_part[0],
              s1.max_value_bound_per_part[1] + s2.max_value_bound_per_part[1]]
    result = U48(parts, bounds)
    return result.normalize() if result.needs_normalize() else result

  def __radd__(self, other):
    if other == 0: return self
    return self + other

  def __sub__(self, other: 'U48') -> 'U48':
    s1 = self.normalize() if self.needs_normalize() else self
    s2 = other.normalize() if other.needs_normalize() else other
    # Assumes s1 >= s2.
    borrow = (s1.parts[0] < s2.parts[0]).astype(jnp.int32)
    p0 = s1.parts[0] - s2.parts[0] + (borrow << 24)
    p1 = s1.parts[1] - s2.parts[1] - borrow
    return U48([p0, p1], s1.max_value_bound_per_part)

  def __mul__(self, other: jax.Array) -> 'U48':
    return U48([p * other for p in self.parts], self.max_value_bound_per_part)

  def __rmul__(self, other: jax.Array) -> 'U48':
    return self.__mul__(other)

  def __lt__(self, other: 'U48') -> jax.Array:
    s1 = self.normalize() if self.needs_normalize() else self
    s2 = other.normalize() if other.needs_normalize() else other
    less_high = s1.parts[1] < s2.parts[1]
    equal_high = s1.parts[1] == s2.parts[1]
    less_low = s1.parts[0] < s2.parts[0]
    return less_high | (equal_high & less_low)

  # Compatibility methods
  def harmonize(self): return self.normalize()
  def needs_harmonize(self): return self.needs_normalize()


# Alias for backward compatibility
HighPrecisionUInt = U48


def modulo_u128_u64(dividend_u32: list[jax.Array], divisor_u32: list[jax.Array]) -> Tuple[jax.Array, jax.Array]:
  """Compute (128-bit dividend) % (64-bit divisor) using only 32-bit operations."""
  bh = divisor_u32[0]
  bl = divisor_u32[1]

  init_state = (jnp.uint32(0), jnp.uint32(0))

  def body_fun(i, state):
    rh, rl = state
    bit_idx = 127 - i
    word_idx = 3 - (bit_idx >> 5) # bit_idx // 32
    bit_in_word = bit_idx & 31 # bit_idx % 32
    bit = (dividend_u32[word_idx] >> bit_in_word) & jnp.uint32(1)

    new_rh = (rh << 1) | (rl >> 31)
    new_rl = (rl << 1) | bit
    overflow = (rh >> 31) & jnp.uint32(1)

    is_greater = (
      (overflow == jnp.uint32(1)) |
      (new_rh > bh) |
      ((new_rh == bh) & (new_rl >= bl))
    )

    borrow = jnp.where(new_rl < bl, jnp.uint32(1), jnp.uint32(0))
    sub_rh = new_rh - bh - borrow
    sub_rl = new_rl - bl

    rh_next = jnp.where(is_greater, sub_rh, new_rh)
    rl_next = jnp.where(is_greater, sub_rl, new_rl)
    return (rh_next, rl_next)

  final_rh, final_rl = jax.lax.fori_loop(0, 128, body_fun, init_state)
  return final_rh, final_rl


def random_u48(
  rng_keys: jax.Array,
  max_val: U48,
  shape: Tuple[int, ...] = (),
) -> U48:
  """Generate random U48 values uniformly in [0, max_val)."""
  max_val_u64_in_u32s = max_val.to_u32_pair()
  random_u128_in_u32s = [jax.random.bits(key, shape=shape, dtype=jnp.uint32) for key in rng_keys]
  sampled_u64_in_u32s = jax.vmap(modulo_u128_u64)(random_u128_in_u32s, max_val_u64_in_u32s)
  return U48.from_u32_pair(sampled_u64_in_u32s)

random_high_precision_uint = random_u48


# Register PyTree
def _u48_tree_flatten(hpu: U48):
  return (hpu.parts, (hpu.max_value_bound_per_part,))

def _u48_tree_unflatten(aux_data, children):
  return U48(children, aux_data[0])

tree_util.register_pytree_node(U48, _u48_tree_flatten, _u48_tree_unflatten)
