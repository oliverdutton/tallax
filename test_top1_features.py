"""
Test suite for top1 branch features: take_along_axis and top1 function
"""

import jax
import jax.numpy as jnp
import numpy as np
from tallax.tax.gather import take_along_axis, pallas_compatible_take_along_axis
from tallax.tax.bitonic_topk import top1
from tallax.utils import NUM_LANES, NUM_SUBLANES

def test_take_along_axis_basic():
    """Test basic take_along_axis functionality"""
    print("\n=== Testing take_along_axis (basic) ===")

    # Test case 1: Simple gather along axis 0
    values = jnp.array([[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0]])
    indices = jnp.array([[0, 2, 1, 0],
                         [1, 0, 2, 1],
                         [2, 1, 0, 2]])

    # Expected result when gathering along axis 0
    # For each column, gather from the row specified by indices
    expected_axis0 = jnp.array([[1.0, 10.0, 7.0, 4.0],
                                [5.0, 2.0, 11.0, 8.0],
                                [9.0, 6.0, 3.0, 12.0]])

    result_axis0 = pallas_compatible_take_along_axis(values, indices, axis=0)
    print(f"Values shape: {values.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"Result axis=0 shape: {result_axis0.shape}")
    print(f"Expected:\n{expected_axis0}")
    print(f"Got:\n{result_axis0}")

    # Use numpy for comparison
    np_result = np.take_along_axis(np.array(values), np.array(indices), axis=0)
    print(f"NumPy result:\n{np_result}")

    # Test case 2: Simple gather along axis 1
    indices_axis1 = jnp.array([[0, 2, 1],
                               [1, 0, 3],
                               [2, 1, 0]])

    expected_axis1 = jnp.array([[1.0, 3.0, 2.0],
                                [6.0, 5.0, 8.0],
                                [11.0, 10.0, 9.0]])

    result_axis1 = pallas_compatible_take_along_axis(values, indices_axis1, axis=1)
    print(f"\nResult axis=1 shape: {result_axis1.shape}")
    print(f"Expected:\n{expected_axis1}")
    print(f"Got:\n{result_axis1}")

    np_result_axis1 = np.take_along_axis(np.array(values), np.array(indices_axis1), axis=1)
    print(f"NumPy result:\n{np_result_axis1}")

    return result_axis0, result_axis1

def test_take_along_axis_with_topk():
    """Test take_along_axis with top-k indices"""
    print("\n=== Testing take_along_axis with top-k indices ===")

    # Create a batch of logits
    batch_size = 4
    vocab_size = 256
    k = 128

    key = jax.random.PRNGKey(42)
    logits = jax.random.normal(key, (batch_size, vocab_size))

    # Get top-k indices using argsort
    topk_indices = jnp.argsort(logits, axis=1)[:, -k:][:, ::-1]

    # Gather top-k logits using take_along_axis
    topk_logits_gathered = pallas_compatible_take_along_axis(logits, topk_indices, axis=1)

    # Compare with jnp.take_along_axis
    topk_logits_expected = jnp.take_along_axis(logits, topk_indices, axis=1)

    print(f"Logits shape: {logits.shape}")
    print(f"Top-k indices shape: {topk_indices.shape}")
    print(f"Gathered logits shape: {topk_logits_gathered.shape}")
    print(f"Max difference from jnp.take_along_axis: {jnp.max(jnp.abs(topk_logits_gathered - topk_logits_expected))}")

    # Check if results match
    matches = jnp.allclose(topk_logits_gathered, topk_logits_expected, rtol=1e-5, atol=1e-5)
    print(f"Results match: {matches}")

    if not matches:
        print(f"First mismatch:")
        diff = jnp.abs(topk_logits_gathered - topk_logits_expected)
        max_diff_idx = jnp.unravel_index(jnp.argmax(diff), diff.shape)
        print(f"  Position: {max_diff_idx}")
        print(f"  Expected: {topk_logits_expected[max_diff_idx]}")
        print(f"  Got: {topk_logits_gathered[max_diff_idx]}")
        print(f"  Difference: {diff[max_diff_idx]}")

    return topk_logits_gathered, topk_logits_expected

def test_top1_basic():
    """Test basic top1 functionality"""
    print("\n=== Testing top1 function (basic) ===")

    # Create test data with power-of-2 size >= NUM_SUBLANES
    size = max(16, NUM_SUBLANES)  # At least 16 or NUM_SUBLANES
    width = 256

    key = jax.random.PRNGKey(42)
    values = jax.random.normal(key, (size, width))
    indices = jnp.arange(width)[None, :].repeat(size, axis=0)

    print(f"Values shape: {values.shape}")
    print(f"Indices shape: {indices.shape}")

    # Run top1
    result_values, result_indices = top1([values, indices], num_keys=1, axis=0)

    print(f"Result values shape: {result_values.shape}")
    print(f"Result indices shape: {result_indices.shape}")

    # Expected: top value along axis 0 for each column
    expected_values = jnp.max(values, axis=0, keepdims=True)
    expected_indices = jnp.argmax(values, axis=0, keepdims=True)

    print(f"Expected values shape: {expected_values.shape}")
    print(f"Expected indices shape: {expected_indices.shape}")
    print(f"Max difference in values: {jnp.max(jnp.abs(result_values - expected_values))}")
    print(f"Indices match: {jnp.all(result_indices == expected_indices)}")

    values_match = jnp.allclose(result_values, expected_values, rtol=1e-5, atol=1e-5)
    indices_match = jnp.all(result_indices == expected_indices)

    print(f"Values match: {values_match}")
    print(f"Indices match: {indices_match}")

    if not values_match or not indices_match:
        print("\nFirst few results:")
        print(f"Expected values: {expected_values[0, :10]}")
        print(f"Got values: {result_values[0, :10]}")
        print(f"Expected indices: {expected_indices[0, :10]}")
        print(f"Got indices: {result_indices[0, :10]}")

    return result_values, result_indices, expected_values, expected_indices

def test_top1_with_gumbel():
    """Test top1 with Gumbel noise (as used in sampling)"""
    print("\n=== Testing top1 with Gumbel noise ===")

    # Simulate the sampling scenario
    batch_size = 16  # Power of 2, >= NUM_SUBLANES
    k = 128

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    # Create logits and indices
    logits = jax.random.normal(key1, (k, batch_size))
    indices = jnp.arange(k)[:, None].repeat(batch_size, axis=1)

    # Add Gumbel noise
    u = jax.random.uniform(key2, logits.shape, minval=1e-10, maxval=1.0)
    gumbel = -jnp.log(-jnp.log(u))
    gumbel_logits = logits + gumbel

    print(f"Gumbel logits shape: {gumbel_logits.shape}")
    print(f"Indices shape: {indices.shape}")

    # Get top1
    result_values, result_indices = top1([gumbel_logits, indices], num_keys=1, axis=0)

    print(f"Result shape: {result_values.shape}")
    print(f"Sampled indices shape: {result_indices.shape}")

    # Verify we got one sample per batch element
    print(f"Result values: {result_values.squeeze()}")
    print(f"Sampled indices: {result_indices.squeeze()}")

    # Check that sampled indices are in valid range
    valid_range = jnp.all((result_indices >= 0) & (result_indices < k))
    print(f"All sampled indices in valid range [0, {k}): {valid_range}")

    return result_indices

def test_pallas_kernel():
    """Test the full Pallas kernel version"""
    print("\n=== Testing take_along_axis Pallas kernel ===")

    try:
        batch_size = 8
        vocab_size = 256
        k = 64

        key = jax.random.PRNGKey(42)
        values = jax.random.normal(key, (batch_size, vocab_size))
        indices = jax.random.randint(key, (batch_size, k), 0, vocab_size)

        print(f"Values shape: {values.shape}")
        print(f"Indices shape: {indices.shape}")

        # Test with interpret=True (CPU mode)
        result = take_along_axis(values, indices, axis=1, interpret=True)
        expected = jnp.take_along_axis(values, indices, axis=1)

        print(f"Result shape: {result.shape}")
        print(f"Max difference: {jnp.max(jnp.abs(result - expected))}")
        print(f"Results match: {jnp.allclose(result, expected, rtol=1e-5, atol=1e-5)}")

        return result, expected
    except Exception as e:
        print(f"Error testing Pallas kernel: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    print("=" * 70)
    print("Testing top1 branch features")
    print("=" * 70)

    # Test 1: Basic take_along_axis
    try:
        test_take_along_axis_basic()
    except Exception as e:
        print(f"ERROR in test_take_along_axis_basic: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: take_along_axis with top-k
    try:
        test_take_along_axis_with_topk()
    except Exception as e:
        print(f"ERROR in test_take_along_axis_with_topk: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Basic top1
    try:
        test_top1_basic()
    except Exception as e:
        print(f"ERROR in test_top1_basic: {e}")
        import traceback
        traceback.print_exc()

    # Test 4: top1 with Gumbel
    try:
        test_top1_with_gumbel()
    except Exception as e:
        print(f"ERROR in test_top1_with_gumbel: {e}")
        import traceback
        traceback.print_exc()

    # Test 5: Pallas kernel
    try:
        test_pallas_kernel()
    except Exception as e:
        print(f"ERROR in test_pallas_kernel: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Testing complete")
    print("=" * 70)
