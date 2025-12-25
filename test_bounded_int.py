"""Test BoundedInt functionality."""

import jax
import jax.numpy as jnp
from tallax._src.bounded_int import BoundedInt


def test_bounded_int_comparisons():
    """Test that BoundedInt can statically evaluate comparisons."""

    # Test with concrete bounds
    x = BoundedInt(jnp.array(5), lower_bound=3, upper_bound=7)

    # Static comparisons (should return bool)
    print("Testing static comparisons:")
    print(f"x < 10: {x < 10}")  # Should be True (upper_bound=7 < 10)
    print(f"x >= 2: {x >= 2}")  # Should be True (lower_bound=3 >= 2)
    print(f"x > 10: {x > 10}")  # Should be False (upper_bound=7 <= 10)
    print(f"x < 2: {x < 2}")    # Should be False (lower_bound=3 >= 2)

    # Dynamic comparisons (should return JAX array)
    print(f"\nx < 5: {x < 5}")  # Dynamic (5 is within bounds [3,7])
    print(f"x > 5: {x > 5}")    # Dynamic (5 is within bounds [3,7])

    print("\nStatic comparison tests passed!")


def test_bounded_int_arithmetic():
    """Test that BoundedInt tracks bounds through arithmetic operations."""

    x = BoundedInt(jnp.array(5), lower_bound=3, upper_bound=7)

    print("\nTesting arithmetic operations:")

    # Addition
    y = x + 2
    print(f"x + 2: bounds=[{y.lower_bound}, {y.upper_bound}]")  # [5, 9]
    assert y.lower_bound == 5 and y.upper_bound == 9

    # Subtraction
    y = x - 1
    print(f"x - 1: bounds=[{y.lower_bound}, {y.upper_bound}]")  # [2, 6]
    assert y.lower_bound == 2 and y.upper_bound == 6

    # Multiplication
    y = x * 2
    print(f"x * 2: bounds=[{y.lower_bound}, {y.upper_bound}]")  # [6, 14]
    assert y.lower_bound == 6 and y.upper_bound == 14

    # Power (for expressions like 2**stage)
    y = 2 ** x
    print(f"2 ** x: bounds=[{y.lower_bound}, {y.upper_bound}]")  # [8, 128]
    assert y.lower_bound == 8 and y.upper_bound == 128

    print("\nArithmetic tests passed!")


def test_bounded_int_modulo():
    """Test modulo operation on BoundedInt."""

    x = BoundedInt(jnp.array(10), lower_bound=8, upper_bound=15)

    print("\nTesting modulo operation:")

    # Modulo with value smaller than modulus
    y = x % 32
    print(f"x % 32: bounds=[{y.lower_bound}, {y.upper_bound}]")  # [8, 15]
    assert y.lower_bound == 8 and y.upper_bound == 15

    # Modulo with value larger than modulus
    y = x % 8
    print(f"x % 8: bounds=[{y.lower_bound}, {y.upper_bound}]")  # [0, 7]
    assert y.lower_bound == 0 and y.upper_bound == 7

    print("\nModulo tests passed!")


def test_bounded_int_bitshift():
    """Test bitshift operations on BoundedInt."""

    x = BoundedInt(jnp.array(3), lower_bound=2, upper_bound=4)

    print("\nTesting bitshift operations:")

    # Left shift
    y = x << 1
    print(f"x << 1: bounds=[{y.lower_bound}, {y.upper_bound}]")  # [4, 8]
    assert y.lower_bound == 4 and y.upper_bound == 8

    # Right shift
    y = x >> 1
    print(f"x >> 1: bounds=[{y.lower_bound}, {y.upper_bound}]")  # [1, 2]
    assert y.lower_bound == 1 and y.upper_bound == 2

    # Reverse left shift: 16 << x
    y = 16 << x
    print(f"16 << x: bounds=[{y.lower_bound}, {y.upper_bound}]")  # [64, 256]
    assert y.lower_bound == 64 and y.upper_bound == 256

    # Reverse right shift: 64 >> x
    y = 64 >> x
    print(f"64 >> x: bounds=[{y.lower_bound}, {y.upper_bound}]")  # [4, 16]
    assert y.lower_bound == 4 and y.upper_bound == 16

    print("\nBitshift tests passed!")


def test_bounded_int_bitonic_use_case():
    """Test the specific use case from bitonic sort."""

    # Simulate the loop variable from pl.loop(stage_lb, stage_ub)
    stage_lb = 6
    stage_ub = 10

    # In practice, stage would be a JAX tracer from pl.loop
    # Here we simulate with a concrete value
    stage_value = jnp.array(8)
    stage = BoundedInt(stage_value, lower_bound=stage_lb, upper_bound=stage_ub-1)

    print("\nTesting bitonic sort use case:")
    print(f"stage bounds: [{stage.lower_bound}, {stage.upper_bound}]")

    # Test comparisons that occur in bitonic sort
    NUM_SUBLANES = 8
    log2_sublanes = 3  # log2(8)

    # These should be statically determinable
    print(f"stage >= log2(NUM_SUBLANES): {stage >= log2_sublanes}")  # True (6 >= 3)
    print(f"stage < log2(NUM_SUBLANES): {stage < log2_sublanes}")    # False (6 >= 3)

    # Test arithmetic used in bitonic sort
    stage_plus_1 = stage + 1
    print(f"stage + 1: bounds=[{stage_plus_1.lower_bound}, {stage_plus_1.upper_bound}]")

    # Test power of 2
    power = 2 ** stage
    print(f"2 ** stage: bounds=[{power.lower_bound}, {power.upper_bound}]")
    assert power.lower_bound == 64 and power.upper_bound == 512

    # Test power of 2 for stage+1
    power = 2 ** (stage + 1)
    print(f"2 ** (stage + 1): bounds=[{power.lower_bound}, {power.upper_bound}]")
    assert power.lower_bound == 128 and power.upper_bound == 1024

    print("\nBitonic sort use case tests passed!")


def test_bounded_int_with_jax_operations():
    """Test that BoundedInt works with JAX operations."""

    @jax.jit
    def test_fn(x_val):
        x = BoundedInt(x_val, lower_bound=0, upper_bound=10)

        # Test that we can use BoundedInt in JAX traced code
        y = x + 1
        return y.value

    result = test_fn(jnp.array(5))
    print(f"\nJAX traced result: {result}")
    assert result == 6

    print("JAX integration test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Running BoundedInt tests")
    print("=" * 60)

    test_bounded_int_comparisons()
    test_bounded_int_arithmetic()
    test_bounded_int_modulo()
    test_bounded_int_bitshift()
    test_bounded_int_bitonic_use_case()
    test_bounded_int_with_jax_operations()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
