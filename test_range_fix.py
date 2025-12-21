#!/usr/bin/env python3
"""Test with corrected range."""

import jax
import jax.numpy as jnp
from tallax._src.bitonic_topk import bitonic_sort_arrays

def test():
    test_cases = [(8, 128), (8, 256), (8, 2048), (128, 256)]

    for shape in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing {shape}")
        print('='*50)

        key = jax.random.PRNGKey(hash(shape) % 2**32)
        arr = jax.random.randint(key, shape, 0, 100, dtype=jnp.int32)

        result = bitonic_sort_arrays([arr], num_keys=1, descending=False)
        expected = jnp.sort(arr, axis=1)

        is_sorted = all(jnp.all(result[0][i, :-1] <= result[0][i, 1:])
                       for i in range(shape[0]))
        matches = jnp.allclose(result[0], expected)

        print(f"Is sorted: {is_sorted}")
        print(f"Matches: {matches}")
        print(f"Result: {'✓ PASS' if matches else '✗ FAIL'}")

        if not matches:
            diff = result[0] != expected
            if jnp.any(diff):
                idx = jnp.argmax(diff.ravel())
                row, col = idx // shape[1], idx % shape[1]
                print(f"First mismatch [{row},{col}]: got {result[0][row,col]}, expected {expected[row,col]}")

if __name__ == "__main__":
    test()
