#!/usr/bin/env python3
"""Complete test for bitonic sort."""

import jax
import jax.numpy as jnp
from tallax._src.bitonic_topk import bitonic_sort_arrays

def test_complete():
    """Test all key shapes with both ascending and descending."""
    test_cases = [
        (8, 16), (8, 64), (8, 128),  # Small
        (8, 256), (8, 512), (8, 1024), (8, 2048),  # Cross-lane
        (16, 128), (32, 128), (128, 128), (128, 256),  # Various batch sizes
    ]

    all_passed = True
    for shape in test_cases:
        for descending in [False, True]:
            key = jax.random.PRNGKey(hash((shape, descending)) % 2**32)
            arr = jax.random.randint(key, shape, 0, 100, dtype=jnp.int32)

            result = bitonic_sort_arrays([arr], num_keys=1, descending=descending)
            expected = jnp.sort(arr, axis=1)
            if descending:
                expected = expected[:, ::-1]

            matches = jnp.allclose(result[0], expected)

            status = "✓" if matches else "✗"
            desc_str = "desc" if descending else "asc"
            print(f"{status} {shape} {desc_str}")

            if not matches:
                all_passed = False
                diff = result[0] != expected
                if jnp.any(diff):
                    idx = jnp.argmax(diff.ravel())
                    row, col = idx // shape[1], idx % shape[1]
                    print(f"  FAIL at [{row},{col}]: got {result[0][row,col]}, expected {expected[row,col]}")

    print(f"\n{'='*50}")
    if all_passed:
        print("ALL TESTS PASSED! ✓✓✓")
    else:
        print("SOME TESTS FAILED")
    return all_passed

if __name__ == "__main__":
    import sys
    success = test_complete()
    sys.exit(0 if success else 1)
