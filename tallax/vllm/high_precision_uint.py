"""High-precision 48-bit unsigned integer arithmetic using two i32 parts.

Specialized for 48-bit (24 bits per part) for efficiency and deterministic
behavior across platforms.

Parts are stored in descending order: [high, low].
"""

from dataclasses import dataclass
from typing import Tuple, List, Union
import jax
import jax.numpy as jnp
from jax import tree_util
from jax.experimental import enable_x64
from jax.extend.random import threefry_2x32
from jax.experimental import pallas as pl

from tallax.tax.sparse_random import sparse_random_bits
from tallax.tax.utils import NUM_LANES, map_reduce


@dataclass(init=False)
class U48:
  """48-bit unsigned integer using two i32 parts (24 bits each).

  Value = parts[0] * 2^24 + parts[1]

  Attributes:
    parts: List of 2 i32 arrays [high, low] representing the value
    max_value_bound_per_part: List of 2 upper bounds [high_bound, low_bound]
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
      # Constructed from existing parts [high, low]
      self.parts = list(x_or_parts)
      if isinstance(max_val, (list, tuple)):
        self.max_value_bound_per_part = list(max_val)
      else:
        # Fallback if only single max_val provided for parts
        mask = (1 << 24) - 1
        self.max_value_bound_per_part = [
          int((max_val or (2**48 - 1)) >> 24),
          mask,
        ]
    else:
      # Initialization from array and max value
      if max_val is None:
        max_val = 2**48 - 1

      mask = (1 << 24) - 1
      self.parts = [x_or_parts >> 24, x_or_parts & mask]

      if isinstance(max_val, (list, tuple)):
        self.max_value_bound_per_part = list(max_val)
      else:
        self.max_value_bound_per_part = [int(max_val >> 24), mask]

  @classmethod
  def from_i32_array(cls, x: jax.Array, max_val: int, **kwargs) -> "U48":
    """Create from i32 array by splitting into 24-bit parts [high, low]."""
    mask = (1 << 24) - 1
    parts = [x >> 24, x & mask]
    bounds = [int(max_val >> 24), mask]
    return cls(parts, bounds)

  @classmethod
  def from_f32(cls, x: jax.Array, max_val: int, **kwargs) -> "U48":
    """Create from f32 by extracting 24-bit parts [high, low]."""
    mask = (1 << 24) - 1
    modulo = float(1 << 24)
    parts = [
      jnp.floor(x / modulo).astype(jnp.int32),
      jnp.fmod(x, modulo).astype(jnp.int32),
    ]
    bounds = [int(max_val >> 24), mask]
    return cls(parts, bounds)

  def to_f32(self) -> jax.Array:
    """Convert back to f32."""
    base = float(1 << 24)
    return self.parts[0].astype(jnp.float32) * base + self.parts[1].astype(
      jnp.float32
    )

  def to_u64_in_u32s(self) -> Tuple[jax.Array, jax.Array]:
    """Convert to a pair of u32 values (high, low) representing a 64-bit value."""
    normalized = self.normalize() if self.needs_normalize() else self
    # parts[0] = high 24 bits, parts[1] = low 24 bits
    # low 32 bits: bits 0-23 from parts[1], bits 24-31 from bits 0-7 of parts[0]
    low = normalized.parts[1].astype(jnp.uint32) | (
      (normalized.parts[0].astype(jnp.uint32) & 0xFF) << 24
    )
    # high 32 bits: bits 32-47 from bits 8-23 of parts[0]
    high = normalized.parts[0].astype(jnp.uint32) >> 8
    return high, low

  @classmethod
  def from_u64_in_u32s(
    cls, u: Union[Tuple[jax.Array, jax.Array], List[jax.Array]], **kwargs
  ) -> "U48":
    """Create from a pair of u32 values [high, low]."""
    high, low = u
    mask24 = (1 << 24) - 1
    p_low = (low & mask24).astype(jnp.int32)
    p_high = ((low >> 24) | (high << 8)).astype(jnp.int32)
    return cls([p_high, p_low], [mask24, mask24])

  def needs_normalize(self) -> bool:
    """Check if carry propagation is needed."""
    mask = (1 << 24) - 1
    # Check if low part (parts[1]) bound exceeds 24 bits
    if self.max_value_bound_per_part[1] > mask:
      return True
    return any(b >= 2**31 for b in self.max_value_bound_per_part)

  def normalize(self) -> "U48":
    """Propagate carries from low part (parts[1]) to high part (parts[0])."""
    mask = (1 << 24) - 1
    carry = self.parts[1] >> 24
    new_parts = [self.parts[0] + carry, self.parts[1] & mask]

    max_carry = self.max_value_bound_per_part[1] >> 24
    new_max_high = self.max_value_bound_per_part[0] + max_carry

    if new_max_high >= 2**31:
      raise ValueError(
        "Value may exceed 2**54, it might overflow so is not allowed."
      )

    return U48(new_parts, [int(new_max_high), mask])

  def sum(self, axis: int = 1, keepdims: bool = True) -> "U48":
    """Sum along specified axis."""
    num_vals = self.parts[0].shape[axis]
    if (
      self.max_value_bound_per_part[0] == 0
      and self.parts[1].shape[1] > NUM_LANES
      and axis == 1
      and self.parts[1].ndim == 2
      and keepdims
    ):
      # Sub i32 already and not too big, can do partial sums in i32 then normalize and do final sum in u48
      return self.map_reduce_sum(
        self.parts[1],
        self.max_value_bound_per_part[1],
      )

    if any(
      bound * num_vals >= 2**31 for bound in self.max_value_bound_per_part
    ):
      raise ValueError(
        "Sum may exceed 2**31, it might overflow so is not allowed, handle in smaller sums with normalizes."
      )
    summed_parts = [p.sum(axis=axis, keepdims=keepdims) for p in self.parts]
    new_bounds = [b * num_vals for b in self.max_value_bound_per_part]
    result = U48(summed_parts, new_bounds)
    return result.normalize() if result.needs_normalize() else result

  @classmethod
  def map_reduce_sum(
    cls, vals: jax.Array, max_val: int, map_fn=lambda x: x
  ) -> "U48":
    """Sum i32 values into U48 with safe chunked reduction to avoid overflow.

    Chunks the reduction so partial sums stay within i32 as long as possible, then converts to U48.

    Args:
      vals: i32 array of shape (batch, vocab_size)
      max_val: Maximum value of any single element (inclusive)
      map_fn: Optional function applied to each chunk before summing
    """
    num_vals = vals.shape[1]
    # max_val is inclusive, so use (max_val + 1) to ensure
    # max_val * safe_reduce_size < 2**31 (i32 range)
    safe_reduce_size = 2**31 // (max_val + 1)
    num_chunks = num_vals // NUM_LANES
    num_parallel = max(8, pl.cdiv(num_chunks, safe_reduce_size))
    chunks_per_parallel = pl.cdiv(num_chunks, num_parallel)
    actual_partial_sum_max = max_val * chunks_per_parallel

    return map_reduce(
      vals,
      map_fn=map_fn,
      reduce_fn="sum",
      num_parallel=num_parallel,
      apply_post_partial_sums_fn=lambda x: cls.from_i32_array(
        x, max_val=actual_partial_sum_max
      ),
    )

  def __add__(self, other: "U48") -> "U48":
    s1 = self.normalize() if self.needs_normalize() else self
    s2 = other.normalize() if other.needs_normalize() else other
    parts = [s1.parts[0] + s2.parts[0], s1.parts[1] + s2.parts[1]]
    bounds = [
      s1.max_value_bound_per_part[0] + s2.max_value_bound_per_part[0],
      s1.max_value_bound_per_part[1] + s2.max_value_bound_per_part[1],
    ]
    result = U48(parts, bounds)
    return result.normalize() if result.needs_normalize() else result

  def __radd__(self, other):
    if other == 0:
      return self
    return self + other

  def __sub__(self, other: "U48") -> "U48":
    s1 = self.normalize() if self.needs_normalize() else self
    s2 = other.normalize() if other.needs_normalize() else other
    # Assumes s1 >= s2. Borrow from high (parts[0]) when low (parts[1]) underflows.
    borrow = (s1.parts[1] < s2.parts[1]).astype(jnp.int32)
    p_low = s1.parts[1] - s2.parts[1] + (borrow << 24)
    p_high = s1.parts[0] - s2.parts[0] - borrow
    return U48([p_high, p_low], s1.max_value_bound_per_part)

  def __mul__(self, other: jax.Array) -> "U48":
    return U48([p * other for p in self.parts], self.max_value_bound_per_part)

  def __rmul__(self, other: jax.Array) -> "U48":
    return self.__mul__(other)

  def __lt__(self, other: "U48") -> jax.Array:
    s1 = self.normalize() if self.needs_normalize() else self
    s2 = other.normalize() if other.needs_normalize() else other
    less_high = s1.parts[0] < s2.parts[0]
    equal_high = s1.parts[0] == s2.parts[0]
    less_low = s1.parts[1] < s2.parts[1]
    return less_high | (equal_high & less_low)


def sample_random_u128_in_u32s(
  key: jax.Array, shape: tuple[int, ...]
) -> tuple[jax.Array, ...]:
  """Generate a random u128 as 4 u32 arrays from a single JAX RNG key."""
  return tuple(jax.random.bits(key, (4, *shape), jnp.uint32))


def modulo_u128_u64(
  dividend_u32: list[jax.Array], divisor_u32: list[jax.Array]
) -> Tuple[jax.Array, jax.Array]:
  """Compute (128-bit dividend) % (64-bit divisor) using only 32-bit operations."""
  assert (
    len(dividend_u32) == 4
  ), "Dividend should be u128 represented as four u32 arrays"
  assert (
    len(divisor_u32) == 2
  ), "Divisor should be u64 represented as two u32 arrays"
  bh = divisor_u32[0]
  bl = divisor_u32[1]

  init_state = (jnp.zeros_like(dividend_u32[0]),) * 2

  def body_fun(i, state):
    rh, rl = state
    bit_idx = 127 - i
    word_idx = 3 - (bit_idx >> 5)  # bit_idx // 32
    bit_in_word = bit_idx & 31  # bit_idx % 32
    bit = (dividend_u32[word_idx] >> bit_in_word) & jnp.uint32(1)

    new_rh = (rh << 1) | (rl >> 31)
    new_rl = (rl << 1) | bit
    overflow = (rh >> 31) & jnp.uint32(1)

    is_greater = (
      (overflow == jnp.uint32(1))
      | (new_rh > bh)
      | ((new_rh == bh) & (new_rl >= bl))
    )

    borrow = jnp.where(new_rl < bl, jnp.uint32(1), jnp.uint32(0))
    sub_rh = new_rh - bh - borrow
    sub_rl = new_rl - bl

    rh_next = jnp.where(is_greater, sub_rh, new_rh)
    rl_next = jnp.where(is_greater, sub_rl, new_rl)
    return (rh_next, rl_next)

  state = init_state
  for i in range(128):
    state = body_fun(i, state)
  return state
  # final_rh, final_rl = jax.lax.fori_loop(0, 128, body_fun, init_state, unroll=True)
  # return final_rh, final_rl


def random_u48(
  rng_key_refs,
  max_val: U48,
  dim0_indices: jax.Array,
) -> U48:
  """Generate random U48 values uniformly in [0, max_val)."""
  max_val_u64_in_u32s = max_val.to_u64_in_u32s()
  indices = (dim0_indices, jnp.zeros_like(dim0_indices))
  random_u128_in_u32s = [
    sparse_random_bits(key_ref, indices, dim1_size=1)
    for key_ref in rng_key_refs
  ]
  assert len(random_u128_in_u32s) == 4
  sampled_u64_in_u32s = modulo_u128_u64(
    random_u128_in_u32s, max_val_u64_in_u32s
  )
  return U48.from_u64_in_u32s(sampled_u64_in_u32s)


# Register PyTree
def _u48_tree_flatten(val: U48):
  return (val.parts, (val.max_value_bound_per_part,))


def _u48_tree_unflatten(aux_data, children):
  return U48(children, aux_data[0])


tree_util.register_pytree_node(U48, _u48_tree_flatten, _u48_tree_unflatten)
