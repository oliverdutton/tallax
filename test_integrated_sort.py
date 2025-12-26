#!/usr/bin/env python3
"""Test script for integrated bitonic sort in _sort_in_vmem_bitonic."""

import jax
import jax.numpy as jnp
from tallax._src.sort import _sort_in_vmem_bitonic, _sort_in_vmem
from tallax._src.utils import is_cpu_platform, pad

def test_integrated_sort(shape):
    """Test integrated bitonic sort implementation."""
    print(f"\nTesting shape: {shape}")
    interpret = is_cpu_platform()

    # Create test data
    key = jax.random.key(0)
    logits = jax.random.normal(key, shape, dtype=jnp.float32).astype(jnp.bfloat16)

    # Pad to power of 2 as required by the sort function
    from tallax._src.utils import NUM_SUBLANES, NUM_LANES
    padded_logits = pad(logits, (NUM_SUBLANES, 'power_of_2_lanes'), prepend=(False, False))

    print(f"Original shape: {shape}, Padded shape: {padded_logits.shape}")

    # Run integrated bitonic sort
    print(f"Running _sort_in_vmem_bitonic with interpret={interpret}...")
    result_bitonic = _sort_in_vmem_bitonic(
        padded_logits,
        num_keys=1,
        descending=False,
        interpret=interpret,
        stage_unroll=6,
        unroll_stages=True,
    )

    # Unpad results
    result_bitonic_unpadded = result_bitonic[0][:shape[0], :shape[1]]

    # Verify against JAX sort
    expected = jax.lax.sort(logits)

    # Check if results match
    matches_jax = jnp.allclose(result_bitonic_unpadded, expected, rtol=1e-3, atol=1e-3)

    print(f"Bitonic sort matches JAX sort: {matches_jax}")

    if not matches_jax:
        diff_jax = jnp.abs(result_bitonic_unpadded - expected)
        print(f"  Max diff vs JAX: {jnp.max(diff_jax):.6f}")
        print(f"  Mean diff vs JAX: {jnp.mean(diff_jax):.6f}")

    return matches_jax

if __name__ == "__main__":
    print("Testing integrated bitonic sort in _sort_in_vmem...")

    # Test with (16, 256)
    success_256 = test_integrated_sort((16, 256))

    # Test with (16, 1024)
    success_1024 = test_integrated_sort((16, 1024))

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
