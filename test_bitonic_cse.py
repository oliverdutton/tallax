"""Test bitonic_sort with CSE on (8, 1024) input."""

import jax
import jax.numpy as jnp
import time
from tallax._src.bitonic_sort import bitonic_sort

def test_bitonic_sort_cse():
    """Test bitonic_sort with and without CSE on (8, 1024) arrays."""

    # Create test input
    shape = (8, 1024)
    print(f"\n{'='*80}")
    print(f"Testing bitonic_sort with CSE on shape {shape}")
    print(f"{'='*80}\n")

    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, shape, dtype=jnp.float32)

    print(f"Input shape: {x.shape}")
    print(f"Input dtype: {x.dtype}")
    print(f"Sample values: {x[0, :5]}")

    # Test 1: Without CSE
    print(f"\n{'-'*80}")
    print("Test 1: bitonic_sort WITHOUT CSE (interpret mode)")
    print(f"{'-'*80}")

    result_no_cse = bitonic_sort(x, descending=False, apply_cse=False, interpret=True)
    result_no_cse = result_no_cse[0]  # Extract from tuple

    print(f"Output shape: {result_no_cse.shape}")
    print(f"First row (should be sorted): {result_no_cse[0, :10]}")

    # Verify it's sorted
    is_sorted_no_cse = jnp.all(result_no_cse[:, :-1] <= result_no_cse[:, 1:])
    print(f"Is sorted: {is_sorted_no_cse}")

    # Test 2: With CSE
    print(f"\n{'-'*80}")
    print("Test 2: bitonic_sort WITH CSE (interpret mode)")
    print(f"{'-'*80}")

    result_with_cse = bitonic_sort(x, descending=False, apply_cse=True, interpret=True)
    result_with_cse = result_with_cse[0]  # Extract from tuple

    print(f"Output shape: {result_with_cse.shape}")
    print(f"First row (should be sorted): {result_with_cse[0, :10]}")

    # Verify it's sorted
    is_sorted_with_cse = jnp.all(result_with_cse[:, :-1] <= result_with_cse[:, 1:])
    print(f"Is sorted: {is_sorted_with_cse}")

    # Test 3: Compare results
    print(f"\n{'-'*80}")
    print("Test 3: Comparing results")
    print(f"{'-'*80}")

    results_match = jnp.allclose(result_no_cse, result_with_cse, rtol=1e-5, atol=1e-5)
    print(f"Results match: {results_match}")

    if results_match:
        print("\n✓ SUCCESS: CSE produces identical results!")
    else:
        print("\n✗ FAILURE: Results differ!")
        max_diff = jnp.max(jnp.abs(result_no_cse - result_with_cse))
        print(f"  Max difference: {max_diff}")

        # Find where they differ
        diff_mask = jnp.abs(result_no_cse - result_with_cse) > 1e-5
        if jnp.any(diff_mask):
            diff_indices = jnp.argwhere(diff_mask)
            print(f"  First difference at: {diff_indices[0]}")
            idx = tuple(diff_indices[0])
            print(f"  No CSE value: {result_no_cse[idx]}")
            print(f"  With CSE value: {result_with_cse[idx]}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Shape: {shape}")
    print(f"Both versions sorted correctly: {is_sorted_no_cse and is_sorted_with_cse}")
    print(f"Results match: {results_match}")
    print(f"{'='*80}\n")

    return results_match and is_sorted_no_cse and is_sorted_with_cse


if __name__ == "__main__":
    success = test_bitonic_sort_cse()
    exit(0 if success else 1)
