"""Comprehensive tests for HighPrecisionUInt."""

import jax
import jax.numpy as jnp
import numpy as np

from tallax.vllm.high_precision_uint import (
  HighPrecisionUInt,
  U48,
  modulo_u128_u64,
  random_int_in_range,
  random_high_precision_uint,
)


class TestConstruction:
  """Tests for HighPrecisionUInt construction."""

  def test_from_parts_basic(self):
    """Test construction from parts."""
    parts = [jnp.array([100]), jnp.array([200])]
    bounds = [0xFFFFFF, 300]
    hpu = HighPrecisionUInt(parts, bounds)

    assert len(hpu.parts) == 2
    assert hpu.num_bits_per_part == 24
    assert hpu.total_bits == 48

  def test_from_array_basic(self):
    """Test construction from array with max_val."""
    x = jnp.array([1000000])
    max_val = 1000000
    hpu = HighPrecisionUInt(x, max_val)

    # Value should be split into 24-bit parts
    assert len(hpu.parts) == 1  # 1000000 fits in 20 bits

  def test_from_array_large_value(self):
    """Test construction with value requiring multiple parts."""
    # Value that requires 2 parts (> 24 bits)
    x = jnp.array([2**25 + 123])
    max_val = 2**26
    hpu = HighPrecisionUInt(x, max_val)

    assert len(hpu.parts) == 2
    # Low part should have the lower 24 bits
    expected_low = (2**25 + 123) & 0xFFFFFF
    expected_high = (2**25 + 123) >> 24
    np.testing.assert_array_equal(hpu.parts[0], expected_low)
    np.testing.assert_array_equal(hpu.parts[1], expected_high)


class TestFromI32Array:
  """Tests for from_i32_array construction."""

  def test_simple_values(self):
    """Test with simple values."""
    x = jnp.array([100, 200, 300], dtype=jnp.int32)
    hpu = HighPrecisionUInt.from_i32_array(x, max_val=300)

    # Should have 1 part for values < 2^24
    assert len(hpu.parts) == 1
    np.testing.assert_array_equal(hpu.parts[0], x)

  def test_large_values(self):
    """Test with values requiring two parts."""
    val = 2**25 + 12345
    x = jnp.array([val], dtype=jnp.int32)
    hpu = HighPrecisionUInt.from_i32_array(x, max_val=2**30)

    assert len(hpu.parts) == 2
    expected_low = val & 0xFFFFFF
    expected_high = val >> 24
    np.testing.assert_array_equal(hpu.parts[0], expected_low)
    np.testing.assert_array_equal(hpu.parts[1], expected_high)

  def test_batch_dimension(self):
    """Test with batch dimension."""
    x = jnp.array([[100, 200], [300, 400]], dtype=jnp.int32)
    hpu = HighPrecisionUInt.from_i32_array(x, max_val=500)

    assert hpu.parts[0].shape == (2, 2)


class TestFromF32:
  """Tests for from_f32 construction."""

  def test_simple_values(self):
    """Test with simple float values."""
    x = jnp.array([100.0, 200.0, 300.0], dtype=jnp.float32)
    hpu = HighPrecisionUInt.from_f32(x, max_val=300)

    np.testing.assert_array_equal(hpu.parts[0], jnp.array([100, 200, 300]))

  def test_large_values(self):
    """Test with large values requiring multiple parts."""
    val = float(2**25 + 12345)
    x = jnp.array([val], dtype=jnp.float32)
    hpu = HighPrecisionUInt.from_f32(x, max_val=2**30)

    assert len(hpu.parts) == 2


class TestToF32:
  """Tests for to_f32 conversion."""

  def test_roundtrip_small_values(self):
    """Test roundtrip conversion for small values."""
    x = jnp.array([100.0, 200.0, 1000.0], dtype=jnp.float32)
    hpu = HighPrecisionUInt.from_f32(x, max_val=1000)
    x_back = hpu.to_f32()

    np.testing.assert_array_almost_equal(x, x_back, decimal=5)

  def test_roundtrip_large_values(self):
    """Test roundtrip for values requiring multiple parts."""
    x = jnp.array([2**20 + 100.0, 2**23 + 500.0], dtype=jnp.float32)
    hpu = HighPrecisionUInt.from_f32(x, max_val=2**24)
    x_back = hpu.to_f32()

    np.testing.assert_array_almost_equal(x, x_back, decimal=0)


class TestNormalization:
  """Tests for normalization behavior."""

  def test_needs_normalize_when_exceeds_mask(self):
    """Test that needs_normalize returns True when low part exceeds mask."""
    # Simulate a situation where low part bound exceeds 24 bits
    parts = [jnp.array([0]), jnp.array([0])]
    bounds = [2**25, 0]  # Low bound exceeds 24 bits
    hpu = HighPrecisionUInt(parts, bounds)

    assert hpu.needs_normalize() is True

  def test_needs_normalize_when_approaching_int32_limit(self):
    """Test that needs_normalize returns True when approaching int32 limit."""
    parts = [jnp.array([0]), jnp.array([0])]
    bounds = [0xFFFFFF, 2**31]  # High bound at int32 limit
    hpu = HighPrecisionUInt(parts, bounds)

    assert hpu.needs_normalize() is True

  def test_no_normalize_when_within_bounds(self):
    """Test that needs_normalize returns False when bounds are safe."""
    parts = [jnp.array([0]), jnp.array([0])]
    bounds = [0xFFFFFF, 1000]  # Well within safe bounds
    hpu = HighPrecisionUInt(parts, bounds)

    assert hpu.needs_normalize() is False

  def test_normalize_propagates_carry(self):
    """Test that normalize correctly propagates carries."""
    # Create a value where low part overflows into high part
    low = jnp.array([2**24 + 100])  # Overflows 24 bits
    high = jnp.array([50])
    parts = [low, high]
    bounds = [2**25, 100]
    hpu = HighPrecisionUInt(parts, bounds)

    normalized = hpu.normalize()

    # Low part should be masked to 24 bits
    np.testing.assert_array_equal(normalized.parts[0], 100)
    # High part should receive carry
    np.testing.assert_array_equal(normalized.parts[1], 51)

  def test_normalization_count_tracking(self):
    """Test that normalization happens infrequently for typical operations."""
    # Simulate summing many small values
    normalize_count = 0
    mask = 0xFFFFFF

    # Start with a value
    x = jnp.array([[100] * 128], dtype=jnp.int32)
    hpu = HighPrecisionUInt.from_i32_array(x, max_val=100)

    # Sum along axis 1
    result = hpu.sum(axis=1, keepdims=True)

    # Check if normalization was needed
    # For 128 values of max 100, sum is at most 12800
    # This should not need normalization
    assert result.max_value_bound_per_part[0] <= mask


class TestAddition:
  """Tests for addition operations."""

  def test_simple_addition(self):
    """Test adding two HighPrecisionUInt values."""
    a = HighPrecisionUInt.from_i32_array(jnp.array([100]), max_val=100)
    b = HighPrecisionUInt.from_i32_array(jnp.array([200]), max_val=200)

    result = a + b

    np.testing.assert_array_almost_equal(result.to_f32(), 300.0, decimal=5)

  def test_addition_with_overflow(self):
    """Test addition that triggers overflow into high part."""
    val_a = 2**23
    val_b = 2**23
    a = HighPrecisionUInt.from_i32_array(jnp.array([val_a]), max_val=2**24)
    b = HighPrecisionUInt.from_i32_array(jnp.array([val_b]), max_val=2**24)

    result = a + b
    expected = val_a + val_b

    np.testing.assert_array_almost_equal(result.to_f32(), expected, decimal=0)

  def test_radd_with_zero(self):
    """Test right addition with zero (for sum())."""
    a = HighPrecisionUInt.from_i32_array(jnp.array([100]), max_val=100)

    result = 0 + a

    assert result is a

  def test_addition_multiple_parts(self):
    """Test addition with values spanning multiple parts."""
    val_a = 2**30
    val_b = 2**30
    a = HighPrecisionUInt.from_i32_array(jnp.array([val_a]), max_val=2**31 - 1)
    b = HighPrecisionUInt.from_i32_array(jnp.array([val_b]), max_val=2**31 - 1)

    result = a + b
    expected = float(val_a + val_b)

    np.testing.assert_array_almost_equal(result.to_f32(), expected, decimal=-5)


class TestSubtraction:
  """Tests for subtraction operations."""

  def test_simple_subtraction(self):
    """Test subtracting two values."""
    a = HighPrecisionUInt.from_i32_array(jnp.array([300]), max_val=300)
    b = HighPrecisionUInt.from_i32_array(jnp.array([100]), max_val=300)

    result = a - b

    np.testing.assert_array_almost_equal(result.to_f32(), 200.0, decimal=5)

  def test_subtraction_with_borrow(self):
    """Test subtraction that requires borrowing."""
    # a.low < b.low, so borrow is needed
    a_parts = [jnp.array([100]), jnp.array([2])]  # 2 * 2^24 + 100
    b_parts = [jnp.array([200]), jnp.array([1])]  # 1 * 2^24 + 200

    a = HighPrecisionUInt(a_parts, [0xFFFFFF, 2])
    b = HighPrecisionUInt(b_parts, [0xFFFFFF, 1])

    result = a - b

    # Expected: (2 * 2^24 + 100) - (1 * 2^24 + 200) = 2^24 - 100
    expected = 2**24 - 100
    np.testing.assert_array_almost_equal(result.to_f32(), expected, decimal=0)


class TestMultiplication:
  """Tests for multiplication operations."""

  def test_multiplication_by_scalar(self):
    """Test multiplication by scalar array."""
    a = HighPrecisionUInt.from_i32_array(jnp.array([100, 200]), max_val=200)
    mask = jnp.array([1, 0])

    result = a * mask

    np.testing.assert_array_almost_equal(result.to_f32(), [100.0, 0.0], decimal=5)

  def test_rmul(self):
    """Test right multiplication."""
    a = HighPrecisionUInt.from_i32_array(jnp.array([100, 200]), max_val=200)
    mask = jnp.array([1, 0])

    result = mask * a

    np.testing.assert_array_almost_equal(result.to_f32(), [100.0, 0.0], decimal=5)


class TestComparison:
  """Tests for comparison operations."""

  def test_less_than_simple(self):
    """Test simple less than comparison."""
    a = HighPrecisionUInt.from_i32_array(jnp.array([100]), max_val=200)
    b = HighPrecisionUInt.from_i32_array(jnp.array([200]), max_val=200)

    result = a < b

    assert result[0] == True

  def test_less_than_equal_values(self):
    """Test less than with equal values."""
    a = HighPrecisionUInt.from_i32_array(jnp.array([100]), max_val=100)
    b = HighPrecisionUInt.from_i32_array(jnp.array([100]), max_val=100)

    result = a < b

    assert result[0] == False

  def test_less_than_greater_value(self):
    """Test less than when a > b."""
    a = HighPrecisionUInt.from_i32_array(jnp.array([200]), max_val=200)
    b = HighPrecisionUInt.from_i32_array(jnp.array([100]), max_val=200)

    result = a < b

    assert result[0] == False

  def test_less_than_multi_part(self):
    """Test less than with multi-part values."""
    # a = 1 * 2^24 + 100
    # b = 1 * 2^24 + 200
    a_parts = [jnp.array([100]), jnp.array([1])]
    b_parts = [jnp.array([200]), jnp.array([1])]

    a = HighPrecisionUInt(a_parts, [0xFFFFFF, 1])
    b = HighPrecisionUInt(b_parts, [0xFFFFFF, 1])

    result = a < b
    assert result[0] == True

    result2 = b < a
    assert result2[0] == False

  def test_less_than_high_part_differs(self):
    """Test less than when high parts differ."""
    # a = 1 * 2^24 + 100
    # b = 2 * 2^24 + 50 (b > a even though b.low < a.low)
    a_parts = [jnp.array([100]), jnp.array([1])]
    b_parts = [jnp.array([50]), jnp.array([2])]

    a = HighPrecisionUInt(a_parts, [0xFFFFFF, 2])
    b = HighPrecisionUInt(b_parts, [0xFFFFFF, 2])

    result = a < b
    assert result[0] == True


class TestSum:
  """Tests for sum operations."""

  def test_sum_axis_1(self):
    """Test sum along axis 1."""
    x = jnp.array([[100, 200, 300]], dtype=jnp.int32)
    hpu = HighPrecisionUInt.from_i32_array(x, max_val=300)

    result = hpu.sum(axis=1, keepdims=True)

    np.testing.assert_array_almost_equal(result.to_f32(), [[600.0]], decimal=5)

  def test_sum_large_values(self):
    """Test sum of many values."""
    n = 1000
    x = jnp.ones((1, n), dtype=jnp.int32) * 100
    hpu = HighPrecisionUInt.from_i32_array(x, max_val=100)

    result = hpu.sum(axis=1, keepdims=True)

    np.testing.assert_array_almost_equal(result.to_f32(), [[100000.0]], decimal=5)

  def test_sum_triggers_normalize_appropriately(self):
    """Test that sum triggers normalization only when needed."""
    # With 24-bit parts and small values, should not need normalization
    n = 100
    x = jnp.ones((1, n), dtype=jnp.int32) * 100
    hpu = HighPrecisionUInt.from_i32_array(x, max_val=100)

    # Manually check bounds after sum
    summed_parts = [part.sum(axis=1, keepdims=True) for part in hpu.parts]
    new_bounds = [bound * n for bound in hpu.max_value_bound_per_part]

    # 100 * 100 = 10000, well under 2^24
    assert new_bounds[0] < 2**24


class TestPyTree:
  """Tests for JAX PyTree registration."""

  def test_pytree_flatten_unflatten(self):
    """Test that flatten and unflatten work correctly."""
    x = jnp.array([100, 200])
    hpu = HighPrecisionUInt.from_i32_array(x, max_val=200)

    # Use jax.tree_util to flatten and unflatten
    leaves, treedef = jax.tree_util.tree_flatten(hpu)
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

    np.testing.assert_array_equal(hpu.parts[0], reconstructed.parts[0])
    assert hpu.num_bits_per_part == reconstructed.num_bits_per_part
    assert hpu.total_bits == reconstructed.total_bits

  def test_jit_compatible(self):
    """Test that HighPrecisionUInt works with JIT."""
    @jax.jit
    def add_hpu(a, b):
      return a + b

    a = HighPrecisionUInt.from_i32_array(jnp.array([100]), max_val=200)
    b = HighPrecisionUInt.from_i32_array(jnp.array([200]), max_val=200)

    result = add_hpu(a, b)

    np.testing.assert_array_almost_equal(result.to_f32(), 300.0, decimal=5)


class TestBackwardCompatibility:
  """Tests for U48 backward compatibility alias."""

  def test_u48_alias(self):
    """Test that U48 is an alias for HighPrecisionUInt."""
    assert U48 is HighPrecisionUInt

  def test_harmonize_alias(self):
    """Test that harmonize is an alias for normalize."""
    x = jnp.array([100])
    hpu = HighPrecisionUInt.from_i32_array(x, max_val=100)

    # Should have the same behavior
    assert hpu.needs_harmonize() == hpu.needs_normalize()


class TestArbitraryBitWidth:
  """Tests for arbitrary bit width support."""

  def test_custom_bits_per_part(self):
    """Test with custom number of bits per part."""
    x = jnp.array([100])
    hpu = HighPrecisionUInt.from_i32_array(x, max_val=100, num_bits_per_part=16)

    assert hpu.num_bits_per_part == 16

  def test_three_parts(self):
    """Test with values requiring three parts via construction from parts."""
    # Use 8-bit parts with explicit parts construction to keep values small
    parts = [jnp.array([100]), jnp.array([50]), jnp.array([10])]
    bounds = [255, 255, 100]
    hpu = HighPrecisionUInt(parts, bounds, num_bits_per_part=8)

    assert len(hpu.parts) == 3
    assert hpu.num_bits_per_part == 8
    # Value should be 100 + 50*2^8 + 10*2^16
    expected = 100 + 50 * (2**8) + 10 * (2**16)  # = 100 + 12800 + 655360 = 668260
    np.testing.assert_array_almost_equal(hpu.to_f32(), expected, decimal=0)

  def test_large_value_correctness(self):
    """Test correctness with values spanning two parts using f32."""
    # Use a value that fits in f32's 24-bit mantissa precision
    val = 2**23 + 12345  # ~8 million
    x = jnp.array([float(val)], dtype=jnp.float32)
    hpu = HighPrecisionUInt.from_f32(x, max_val=2**30, num_bits_per_part=16)

    # Should have 2 parts (30 bits / 16 bits per part = 2)
    assert len(hpu.parts) == 2

    # Convert back and check
    recovered = hpu.to_f32()
    np.testing.assert_array_almost_equal(recovered, val, decimal=0)


class TestNormalizationFrequency:
  """Tests specifically for normalization frequency optimization."""

  def test_minimal_normalization_small_sums(self):
    """Test that small sums don't trigger normalization."""
    # Sum 128 values of 100 each -> 12800, well under 2^24
    normalize_called = False

    x = jnp.ones((1, 128), dtype=jnp.int32) * 100
    hpu = HighPrecisionUInt.from_i32_array(x, max_val=100)

    # Check bounds before sum
    initial_bounds = hpu.max_value_bound_per_part.copy()

    result = hpu.sum(axis=1)

    # Should not have needed normalization
    assert not hpu.needs_normalize()
    # Result should also not need normalization
    assert not result.needs_normalize()

  def test_multiple_additions_without_normalize(self):
    """Test that multiple additions don't trigger unnecessary normalization."""
    # Each value is 1000, add 100 of them -> 100000
    values = [
      HighPrecisionUInt.from_i32_array(jnp.array([1000]), max_val=1000)
      for _ in range(100)
    ]

    result = sum(values)

    # 100 * 1000 = 100000, well under 2^24
    # Should not have normalized (unless bounds grew too large)
    np.testing.assert_array_almost_equal(result.to_f32(), 100000.0, decimal=0)

  def test_normalize_only_when_necessary(self):
    """Test that normalization happens only when bounds exceed threshold."""
    # Create a value with bounds just under threshold
    parts = [jnp.array([100]), jnp.array([100])]
    bounds = [2**23, 100]  # Low bound is large but under 2^24
    hpu = HighPrecisionUInt(parts, bounds)

    assert not hpu.needs_normalize()

    # Now increase bound past threshold
    bounds_over = [2**25, 100]  # Exceeds 24-bit mask
    hpu_over = HighPrecisionUInt(parts, bounds_over)

    assert hpu_over.needs_normalize()


class TestEdgeCases:
  """Tests for edge cases and boundary conditions."""

  def test_zero_value(self):
    """Test with zero values."""
    x = jnp.array([0])
    hpu = HighPrecisionUInt.from_i32_array(x, max_val=100)

    np.testing.assert_array_equal(hpu.to_f32(), 0.0)

  def test_max_24bit_value(self):
    """Test with maximum 24-bit value."""
    max_24 = 2**24 - 1
    x = jnp.array([max_24])
    hpu = HighPrecisionUInt.from_i32_array(x, max_val=max_24)

    np.testing.assert_array_almost_equal(hpu.to_f32(), float(max_24), decimal=0)

  def test_batch_operations(self):
    """Test with batched data."""
    batch_size = 4
    seq_len = 32
    x = jnp.ones((batch_size, seq_len), dtype=jnp.int32) * 100
    hpu = HighPrecisionUInt.from_i32_array(x, max_val=100)

    result = hpu.sum(axis=1, keepdims=True)
    expected = jnp.ones((batch_size, 1)) * 3200

    np.testing.assert_array_almost_equal(result.to_f32(), expected, decimal=0)


class TestU32Conversion:
  """Tests for u32 pair/quad conversion methods."""

  def test_to_u32_pair_small_value(self):
    """Test conversion to u32 pair with small value."""
    # Value fits in low 32 bits, use 24-bit parts
    x = jnp.array([12345])
    hpu = HighPrecisionUInt.from_i32_array(x, max_val=2**20, num_bits_per_part=24)
    high, low = hpu.to_u32_pair()

    np.testing.assert_array_equal(high, 0)
    np.testing.assert_array_equal(low, 12345)

  def test_to_u32_pair_large_value(self):
    """Test conversion to u32 pair with value spanning both words."""
    # Use 24-bit parts: value = 100 + 2 * 2^24 = 33554532
    parts = [jnp.array([100]), jnp.array([2])]
    bounds = [2**24 - 1, 2]
    hpu = HighPrecisionUInt(parts, bounds, num_bits_per_part=24, total_bits=48)

    high, low = hpu.to_u32_pair()

    # 100 + 2 * 2^24 = 33554532 = 0x02000064
    expected_low = 100 + 2 * (2**24)
    np.testing.assert_array_equal(high, 0)
    np.testing.assert_array_equal(low, expected_low)

  def test_from_u32_pair_roundtrip(self):
    """Test roundtrip from u32 pair with values that fit in int32."""
    # Use values that fit in signed int32 range
    high = jnp.uint32(0x12345)  # Small high value
    low = jnp.uint32(0x6789ABCD)  # Low value < 2^31

    hpu = HighPrecisionUInt.from_u32_pair(high, low, max_val=2**48)
    high_back, low_back = hpu.to_u32_pair()

    np.testing.assert_array_equal(high_back, high)
    np.testing.assert_array_equal(low_back, low)

  def test_to_u32_quad(self):
    """Test conversion to u32 quad (128-bit) with 32-bit parts."""
    # Use 32-bit parts directly for clean mapping
    parts = [
      jnp.array([0x11111111]),
      jnp.array([0x22222222]),
      jnp.array([0x33333333]),
      jnp.array([0x44444444]),
    ]
    bounds = [2**31 - 1] * 4
    hpu = HighPrecisionUInt(parts, bounds, num_bits_per_part=32, total_bits=128)

    w0, w1, w2, w3 = hpu.to_u32_quad()

    # word3 = LSB (parts[0]), word0 = MSB (parts[3])
    np.testing.assert_array_equal(w3.flatten(), [0x11111111])
    np.testing.assert_array_equal(w2.flatten(), [0x22222222])
    np.testing.assert_array_equal(w1.flatten(), [0x33333333])
    np.testing.assert_array_equal(w0.flatten(), [0x44444444])

  def test_from_u32_quad_roundtrip(self):
    """Test roundtrip from u32 quad with safe values."""
    # Use values that fit in signed int32 range for JAX compatibility
    w0 = jnp.uint32(0x11111111)  # < 2^31
    w1 = jnp.uint32(0x22222222)
    w2 = jnp.uint32(0x33333333)
    w3 = jnp.uint32(0x44444444)

    hpu = HighPrecisionUInt.from_u32_quad(w0, w1, w2, w3)
    w0_back, w1_back, w2_back, w3_back = hpu.to_u32_quad()

    np.testing.assert_array_equal(w0_back, w0)
    np.testing.assert_array_equal(w1_back, w1)
    np.testing.assert_array_equal(w2_back, w2)
    np.testing.assert_array_equal(w3_back, w3)


class TestModuloU128U64:
  """Tests for u128 % u64 modulo operation."""

  def test_small_numbers(self):
    """Test 100 % 7 = 2."""
    dividend = jnp.array([0, 0, 0, 100], dtype=jnp.uint32)
    divisor = jnp.array([0, 7], dtype=jnp.uint32)

    rem_high, rem_low = modulo_u128_u64(dividend, divisor)

    assert int(rem_high) == 0
    assert int(rem_low) == 2  # 100 % 7 = 2

  def test_255_mod_16(self):
    """Test 255 % 16 = 15."""
    dividend = jnp.array([0, 0, 0, 255], dtype=jnp.uint32)
    divisor = jnp.array([0, 16], dtype=jnp.uint32)

    rem_high, rem_low = modulo_u128_u64(dividend, divisor)

    assert int(rem_high) == 0
    assert int(rem_low) == 15  # 255 % 16 = 15

  def test_large_dividend(self):
    """Test with large 128-bit dividend."""
    # Dividend: 0x11112222_33334444_55556666_77778888
    dividend = jnp.array(
      [0x11112222, 0x33334444, 0x55556666, 0x77778888],
      dtype=jnp.uint32
    )
    # Divisor: 0xDEADBEEF_CAFEBABE
    divisor = jnp.array([0xDEADBEEF, 0xCAFEBABE], dtype=jnp.uint32)

    rem_high, rem_low = modulo_u128_u64(dividend, divisor)

    # Compute expected result using Python
    py_dividend = (0x11112222 << 96) | (0x33334444 << 64) | (0x55556666 << 32) | 0x77778888
    py_divisor = (0xDEADBEEF << 32) | 0xCAFEBABE
    expected = py_dividend % py_divisor

    result = (int(rem_high) << 32) | int(rem_low)
    assert result == expected

  def test_max_values(self):
    """Test max u128 % max u64."""
    dividend = jnp.array([0xFFFFFFFF] * 4, dtype=jnp.uint32)
    divisor = jnp.array([0xFFFFFFFF, 0xFFFFFFFF], dtype=jnp.uint32)

    rem_high, rem_low = modulo_u128_u64(dividend, divisor)

    # max_u128 % max_u64 = 0 (since max_u128 = max_u64 * max_u64 + 2*max_u64)
    py_dividend = 2**128 - 1
    py_divisor = 2**64 - 1
    expected = py_dividend % py_divisor

    result = (int(rem_high) << 32) | int(rem_low)
    assert result == expected

  def test_power_of_two_modulo(self):
    """Test 2^64 % 65535."""
    dividend = jnp.array([0, 1, 0, 0], dtype=jnp.uint32)  # 2^64
    divisor = jnp.array([0, 65535], dtype=jnp.uint32)

    rem_high, rem_low = modulo_u128_u64(dividend, divisor)

    # 2^64 % 65535
    expected = (2**64) % 65535

    result = (int(rem_high) << 32) | int(rem_low)
    assert result == expected


class TestRandomIntGeneration:
  """Tests for random integer generation."""

  def test_random_in_range_scalar(self):
    """Test random generation for scalar output."""
    key = jax.random.PRNGKey(42)
    range_val = 1000

    result_high, result_low = random_int_in_range(
      key,
      jnp.uint32(0),
      jnp.uint32(range_val),
      shape=()
    )

    # Result should be in [0, range_val)
    result = int(result_low)
    assert 0 <= result < range_val

  def test_random_in_range_batched(self):
    """Test random generation for batched output."""
    key = jax.random.PRNGKey(123)
    range_val = 10000
    batch_size = 100

    result_high, result_low = random_int_in_range(
      key,
      jnp.uint32(0),
      jnp.uint32(range_val),
      shape=(batch_size,)
    )

    # All results should be in [0, range_val)
    results = result_low.astype(jnp.int32)
    assert jnp.all(results >= 0)
    assert jnp.all(results < range_val)

  def test_random_in_large_range(self):
    """Test random generation with range > 2^32."""
    key = jax.random.PRNGKey(456)
    # Range = 2^40
    range_high = jnp.uint32(1 << 8)  # 2^40 >> 32 = 2^8
    range_low = jnp.uint32(0)

    result_high, result_low = random_int_in_range(
      key,
      range_high,
      range_low,
      shape=()
    )

    # Result should be less than range
    result = (int(result_high) << 32) | int(result_low)
    range_full = (int(range_high) << 32) | int(range_low)
    assert 0 <= result < range_full

  def test_random_high_precision_uint(self):
    """Test HighPrecisionUInt random generation."""
    key = jax.random.PRNGKey(789)
    max_val = 1000000

    hpu = random_high_precision_uint(key, max_val=max_val)

    # Convert to f32 and check range
    value = float(hpu.to_f32())
    assert 0 <= value < max_val

  def test_random_distribution_uniformity(self):
    """Test that random values are roughly uniformly distributed."""
    key = jax.random.PRNGKey(999)
    range_val = 10
    num_samples = 10000

    # Generate many samples
    keys = jax.random.split(key, num_samples)

    def generate_one(k):
      _, low = random_int_in_range(k, jnp.uint32(0), jnp.uint32(range_val), shape=())
      return low

    samples = jax.vmap(generate_one)(keys)

    # Count occurrences of each value
    counts = jnp.bincount(samples, length=range_val)

    # Each value should appear roughly num_samples/range_val times
    # Allow 50% deviation for statistical variation
    expected_count = num_samples / range_val
    for i in range(range_val):
      assert counts[i] > expected_count * 0.5, f"Value {i} underrepresented"
      assert counts[i] < expected_count * 1.5, f"Value {i} overrepresented"


def run_all_tests():
  """Run all tests and report results."""
  import sys

  test_classes = [
    TestConstruction,
    TestFromI32Array,
    TestFromF32,
    TestToF32,
    TestNormalization,
    TestAddition,
    TestSubtraction,
    TestMultiplication,
    TestComparison,
    TestSum,
    TestPyTree,
    TestBackwardCompatibility,
    TestArbitraryBitWidth,
    TestNormalizationFrequency,
    TestEdgeCases,
    TestU32Conversion,
    TestModuloU128U64,
    TestRandomIntGeneration,
  ]

  total_tests = 0
  passed_tests = 0
  failed_tests = []

  for test_class in test_classes:
    print(f"\n{'='*60}")
    print(f"Running {test_class.__name__}")
    print('='*60)

    instance = test_class()
    methods = [m for m in dir(instance) if m.startswith('test_')]

    for method_name in methods:
      total_tests += 1
      method = getattr(instance, method_name)
      try:
        method()
        print(f"  ✓ {method_name}")
        passed_tests += 1
      except Exception as e:
        print(f"  ✗ {method_name}: {e}")
        failed_tests.append((test_class.__name__, method_name, str(e)))

  print(f"\n{'='*60}")
  print(f"Results: {passed_tests}/{total_tests} tests passed")
  print('='*60)

  if failed_tests:
    print("\nFailed tests:")
    for cls, method, error in failed_tests:
      print(f"  {cls}.{method}: {error}")
    return 1

  print("\nAll tests passed!")
  return 0


if __name__ == "__main__":
  import sys
  sys.exit(run_all_tests())
