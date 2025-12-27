#!/usr/bin/env python3
"""Test if segfaults occur after running many tests in sequence."""

import jax
import jax.numpy as jnp
from tallax._src.test_utils import verify_sort_output
import sys

def run_many_tests():
    """Run many tests in sequence to see if accumulated state causes segfault."""

    # All test variants
    variants = [
        ("standard", {"return_argsort": False, "is_stable": False, "descending": False}),
        ("return_argsort", {"return_argsort": True, "is_stable": False, "descending": False}),
        ("stable", {"return_argsort": False, "is_stable": True, "descending": False}),
        ("stable_argsort", {"return_argsort": True, "is_stable": True, "descending": False}),
        ("descending", {"return_argsort": False, "is_stable": False, "descending": True}),
        ("descending_argsort", {"return_argsort": True, "is_stable": False, "descending": True}),
        ("descending_stable", {"return_argsort": False, "is_stable": True, "descending": True}),
        ("descending_stable_argsort", {"return_argsort": True, "is_stable": True, "descending": True}),
    ]

    dtypes = [jnp.bfloat16, jnp.float32]
    sizes = [128, 256]  # Match pytest test selection

    print("="*70)
    print("SEQUENTIAL LOAD TEST")
    print("="*70)
    print(f"Running {len(variants)} variants × {len(dtypes)} dtypes × {len(sizes)} sizes")
    print(f"Total: {len(variants) * len(dtypes) * len(sizes)} tests")
    print("="*70)

    test_count = 0
    passed = 0
    failed = 0

    for size in sizes:
        for dtype in dtypes:
            for variant_name, kwargs in variants:
                test_count += 1
                shape = (16, size)

                # Format test name like pytest
                test_name = f"test_sort_comprehensive[{dtype.__name__}-{size}-{variant_name}]"
                print(f"\n[{test_count:3d}] {test_name}...", end=" ", flush=True)

                key = jax.random.key(0)
                arr = jax.random.normal(key, shape, dtype=jnp.float32).astype(dtype)
                operands = [arr]

                try:
                    verify_sort_output(
                        operands,
                        num_keys=1,
                        interpret=True,
                        **kwargs
                    )
                    print("✓ PASSED")
                    passed += 1
                except Exception as e:
                    print(f"✗ FAILED: {type(e).__name__}")
                    failed += 1
                    # Print first failure details
                    if failed == 1:
                        import traceback
                        print("\nFirst failure traceback:")
                        traceback.print_exc()
                        print()

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total tests run: {test_count}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n✅ All tests passed! No segfault in sequential execution.")
    else:
        print(f"\n⚠️  {failed} tests failed")
        print("Note: If this run completed, there was no segfault")
        print("      (segfaults would have killed the process)")


if __name__ == "__main__":
    run_many_tests()
