#!/usr/bin/env python3
"""Test bitonic sort refactor with SymInt and new kwargs."""

import jax
import jax.numpy as jnp
from tallax._src.bitonic_sort import bitonic_sort


def test_bitonic_sort_basic():
    """Test basic bitonic sort functionality with shape (16, 1024)."""
    print("Testing bitonic sort with shape (16, 1024)...")

    # Create test data
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, shape=(16, 1024), dtype=jnp.float32)

    # Test ascending sort
    print("Testing ascending sort...")
    # Try without interpret mode first to see if it compiles
    result_asc = bitonic_sort(x, descending=False, unroll_stages=True, interpret=True)
    result_asc = result_asc[0]

    # Verify sorting is correct
    for i in range(16):
        row = result_asc[i]
        is_sorted = jnp.all(row[:-1] <= row[1:])
        if not is_sorted:
            print(f"  Row {i}: FAILED - not sorted")
            return False
    print("  Ascending sort: PASSED")

    # Test descending sort
    print("Testing descending sort...")
    result_desc = bitonic_sort(x, descending=True, interpret=True)
    result_desc = result_desc[0]

    # Verify sorting is correct
    for i in range(16):
        row = result_desc[i]
        is_sorted = jnp.all(row[:-1] >= row[1:])
        if not is_sorted:
            print(f"  Row {i}: FAILED - not sorted")
            return False
    print("  Descending sort: PASSED")

    return True


def test_bitonic_sort_with_kwargs():
    """Test bitonic sort with new kwargs."""
    print("\nTesting bitonic sort with kwargs...")

    key = jax.random.PRNGKey(123)
    x = jax.random.uniform(key, shape=(16, 1024), dtype=jnp.float32)

    # Test with max_num_fused_stages
    print("Testing with max_num_fused_stages=5...")
    result = bitonic_sort(x, descending=False, max_num_fused_stages=5, interpret=True)
    result = result[0]
    for i in range(16):
        row = result[i]
        is_sorted = jnp.all(row[:-1] <= row[1:])
        if not is_sorted:
            print(f"  Row {i}: FAILED")
            return False
    print("  max_num_fused_stages=5: PASSED")

    # Test with tile_unroll
    print("Testing with tile_unroll=2...")
    result = bitonic_sort(x, descending=False, tile_unroll=2, interpret=True)
    result = result[0]
    for i in range(16):
        row = result[i]
        is_sorted = jnp.all(row[:-1] <= row[1:])
        if not is_sorted:
            print(f"  Row {i}: FAILED")
            return False
    print("  tile_unroll=2: PASSED")

    # Test with unroll_stages
    print("Testing with unroll_stages=True...")
    result = bitonic_sort(x, descending=False, unroll_stages=True, interpret=True)
    result = result[0]
    for i in range(16):
        row = result[i]
        is_sorted = jnp.all(row[:-1] <= row[1:])
        if not is_sorted:
            print(f"  Row {i}: FAILED")
            return False
    print("  unroll_stages=True: PASSED")

    return True


def test_bitonic_sort_multi_key():
    """Test multi-key sorting."""
    print("\nTesting multi-key sort...")

    key = jax.random.PRNGKey(456)
    x1 = jax.random.randint(key, shape=(16, 1024), minval=0, maxval=10, dtype=jnp.int32)
    x2 = jax.random.uniform(key, shape=(16, 1024), dtype=jnp.float32)

    result = bitonic_sort([x1, x2], num_keys=1, descending=False, interpret=True)
    result_x1, result_x2 = result

    # Verify first key is sorted
    for i in range(16):
        row = result_x1[i]
        is_sorted = jnp.all(row[:-1] <= row[1:])
        if not is_sorted:
            print(f"  Row {i}: FAILED")
            return False
    print("  Multi-key sort: PASSED")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Bitonic Sort Refactor Tests")
    print("=" * 60)

    all_passed = True

    # Run tests
    all_passed &= test_bitonic_sort_basic()
    all_passed &= test_bitonic_sort_with_kwargs()
    all_passed &= test_bitonic_sort_multi_key()

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)
