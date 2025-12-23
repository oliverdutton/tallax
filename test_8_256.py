#!/usr/bin/env python3
"""Test bitonic sort with shape (8, 256) for both unroll_stages settings."""

import jax
import jax.numpy as jnp
from tallax._src.bitonic_sort import bitonic_sort


def test_shape(unroll_stages):
    """Test with shape (8, 256)."""
    print(f"\n{'='*60}")
    print(f"Testing shape (8, 256) with unroll_stages={unroll_stages}")
    print('='*60)

    # Create test data
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, shape=(8, 256), dtype=jnp.float32)

    print("Input sample (first row, first 10 elements):", x[0, :10])

    # Test ascending sort
    print(f"\nTesting ascending sort...")
    try:
        result_asc = bitonic_sort(x, descending=False, unroll_stages=unroll_stages, interpret=True)
        result_asc = result_asc[0]

        print("Output sample (first row, first 10 elements):", result_asc[0, :10])

        # Verify sorting is correct
        all_sorted = True
        for i in range(8):
            row = result_asc[i]
            is_sorted = jnp.all(row[:-1] <= row[1:])
            if not is_sorted:
                print(f"  ❌ Row {i}: FAILED - not sorted")
                all_sorted = False

        if all_sorted:
            print(f"  ✅ All rows sorted correctly!")

        # Compare with expected
        expected = jnp.sort(x, axis=1)
        matches_expected = jnp.allclose(result_asc, expected)
        print(f"  Matches jnp.sort: {matches_expected}")

        return all_sorted and matches_expected

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_descending(unroll_stages):
    """Test descending sort."""
    print(f"\n{'='*60}")
    print(f"Testing DESCENDING sort with unroll_stages={unroll_stages}")
    print('='*60)

    key = jax.random.PRNGKey(123)
    x = jax.random.uniform(key, shape=(8, 256), dtype=jnp.float32)

    print("Input sample (first row, first 10 elements):", x[0, :10])

    try:
        result = bitonic_sort(x, descending=True, unroll_stages=unroll_stages, interpret=True)
        result = result[0]

        print("Output sample (first row, first 10 elements):", result[0, :10])

        # Verify sorting is correct
        all_sorted = True
        for i in range(8):
            row = result[i]
            is_sorted = jnp.all(row[:-1] >= row[1:])
            if not is_sorted:
                print(f"  ❌ Row {i}: FAILED - not sorted descending")
                all_sorted = False

        if all_sorted:
            print(f"  ✅ All rows sorted correctly (descending)!")

        # Compare with expected
        expected = jnp.sort(x, axis=1)[:, ::-1]
        matches_expected = jnp.allclose(result, expected)
        print(f"  Matches jnp.sort (reversed): {matches_expected}")

        return all_sorted and matches_expected

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("BITONIC SORT TEST: Shape (8, 256)")
    print("="*60)

    results = {}

    # Test with unroll_stages=True
    results['unroll_true_asc'] = test_shape(unroll_stages=True)
    results['unroll_true_desc'] = test_descending(unroll_stages=True)

    # Test with unroll_stages=False
    results['unroll_false_asc'] = test_shape(unroll_stages=False)
    results['unroll_false_desc'] = test_descending(unroll_stages=False)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"unroll_stages=True  (ascending):  {'✅ PASS' if results['unroll_true_asc'] else '❌ FAIL'}")
    print(f"unroll_stages=True  (descending): {'✅ PASS' if results['unroll_true_desc'] else '❌ FAIL'}")
    print(f"unroll_stages=False (ascending):  {'✅ PASS' if results['unroll_false_asc'] else '❌ FAIL'}")
    print(f"unroll_stages=False (descending): {'✅ PASS' if results['unroll_false_desc'] else '❌ FAIL'}")
    print("="*60)

    if all(results.values()):
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n⚠️  SOME TESTS FAILED")
