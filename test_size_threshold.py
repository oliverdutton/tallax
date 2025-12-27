#!/usr/bin/env python3
"""Test different sizes to find segfault threshold."""

import jax
import jax.numpy as jnp
from tallax._src.test_utils import verify_sort_output
import sys

def test_size(batch, size, name, **kwargs):
    """Test a specific size and kwarg combination."""
    shape = (batch, size)
    print(f"\nTesting {name} with shape {shape}...", end=" ", flush=True)

    key = jax.random.key(0)
    arr = jax.random.normal(key, shape, dtype=jnp.float32).astype(jnp.bfloat16)
    operands = [arr]

    try:
        verify_sort_output(
            operands,
            num_keys=1,
            interpret=True,
            **kwargs
        )
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}")
        return False


def test_size_progression():
    """Test different sizes to find where segfault starts."""
    # Test configurations that are most likely to trigger issues
    test_configs = [
        ("stable_argsort", {"return_argsort": True, "is_stable": True, "descending": False}),
        ("descending_stable_argsort", {"return_argsort": True, "is_stable": True, "descending": True}),
    ]

    sizes = [128, 256, 512, 1024, 2048]
    batch = 16

    print("="*70)
    print("TESTING SIZE THRESHOLD FOR SEGFAULTS")
    print("="*70)
    print(f"Batch size: {batch}")
    print(f"Testing sizes: {sizes}")
    print()

    results = {}
    for size in sizes:
        print(f"\n{'='*70}")
        print(f"SIZE: {size} (shape: {batch}x{size})")
        print('='*70)

        size_results = {}
        for name, kwargs in test_configs:
            passed = test_size(batch, size, name, **kwargs)
            size_results[name] = passed

        results[size] = size_results

        # Stop if we hit failures
        if not all(size_results.values()):
            print(f"\n⚠️  Failures detected at size {size}, stopping progression")
            break

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for size in sorted(results.keys()):
        all_passed = all(results[size].values())
        status = "✓ ALL PASS" if all_passed else "✗ SOME FAIL"
        print(f"\nSize {size:4d}: {status}")
        for name, passed in results[size].items():
            status = "✓" if passed else "✗"
            print(f"  {status} {name}")

    # Find threshold
    passing_sizes = [s for s in results if all(results[s].values())]
    failing_sizes = [s for s in results if not all(results[s].values())]

    print("\n" + "="*70)
    print("THRESHOLD ANALYSIS")
    print("="*70)
    if passing_sizes and failing_sizes:
        max_passing = max(passing_sizes)
        min_failing = min(failing_sizes)
        print(f"✓ Largest passing size: {max_passing}")
        print(f"✗ Smallest failing size: {min_failing}")
        print(f"\n⚠️  Segfault threshold is between {max_passing} and {min_failing}")
    elif passing_sizes:
        print(f"✓ All tested sizes pass (up to {max(passing_sizes)})")
    else:
        print(f"✗ All sizes fail (starting from {min(results.keys())})")


if __name__ == "__main__":
    test_size_progression()
