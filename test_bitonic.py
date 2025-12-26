#!/usr/bin/env python3
"""Quick test script for bitonic_sort."""

import jax
import jax.numpy as jnp
from tallax._src.bitonic_sort import bitonic_sort
from tallax._src.utils import is_cpu_platform

def test_shape(shape):
    """Test bitonic sort with a specific shape."""
    print(f"\nTesting shape: {shape}")
    interpret = is_cpu_platform()

    # Create test data
    key = jax.random.key(0)
    logits = jax.random.normal(key, shape, dtype=jnp.float32).astype(jnp.bfloat16)

    # Run bitonic sort
    print(f"Running bitonic_sort with interpret={interpret}...")
    result = bitonic_sort(logits, num_keys=1, interpret=interpret)

    # Verify against JAX sort
    expected = jax.lax.sort(logits)

    # Check if results match
    matches = jnp.allclose(result[0], expected, rtol=1e-3, atol=1e-3)
    print(f"Results match JAX sort: {matches}")

    if not matches:
        # Show some differences
        diff = jnp.abs(result[0] - expected)
        max_diff = jnp.max(diff)
        print(f"Max difference: {max_diff}")
        print(f"Mean difference: {jnp.mean(diff)}")

    return matches

if __name__ == "__main__":
    print("Testing bitonic_sort implementation...")

    # Test with (16, 256)
    success_256 = test_shape((16, 256))

    # Test with (16, 1024)
    success_1024 = test_shape((16, 1024))

    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"  (16, 256):  {'PASS' if success_256 else 'FAIL'}")
    print(f"  (16, 1024): {'PASS' if success_1024 else 'FAIL'}")
    print("=" * 50)

    if success_256 and success_1024:
        print("\nAll tests passed!")
        exit(0)
    else:
        print("\nSome tests failed!")
        exit(1)
