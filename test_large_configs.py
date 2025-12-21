#!/usr/bin/env python3
"""Test bitonic sort on large configurations."""

import jax
import jax.numpy as jnp
from tallax._src.bitonic_topk import bitonic_sort_arrays

def test_large_configs():
    """Test bitonic sort on (8, 2048), (16, 4096), (16, 16384)."""

    test_cases = [
        (8, 2048),
        (16, 4096),
        (16, 16384),
    ]

    for shape in test_cases:
        for descending in [False, True]:
            print(f"Testing {shape} {'desc' if descending else 'asc'}...", end=" ", flush=True)

            # Create random test data
            key = jax.random.PRNGKey(42)
            arr = jax.random.randint(key, shape, 0, 10000, dtype=jnp.int32)

            # Sort with bitonic_sort_arrays
            result = bitonic_sort_arrays([arr], num_keys=1, descending=descending)

            # Expected result with JAX sort
            expected = jnp.sort(arr, axis=1)
            if descending:
                expected = expected[:, ::-1]

            # Verify
            if jnp.allclose(result[0], expected):
                print("✓")
            else:
                print("✗ FAILED")
                print(f"  Shape: {shape}")
                print(f"  Descending: {descending}")
                print(f"  First mismatch at:")
                diff = result[0] != expected
                if jnp.any(diff):
                    idx = jnp.argwhere(diff)[0]
                    print(f"    Index: {idx}")
                    print(f"    Got: {result[0][tuple(idx)]}")
                    print(f"    Expected: {expected[tuple(idx)]}")
                return False

    print("\n" + "="*50)
    print("ALL LARGE CONFIG TESTS PASSED! ✓✓✓")
    return True

if __name__ == "__main__":
    test_large_configs()
