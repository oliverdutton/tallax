#!/usr/bin/env python3
"""Investigate segfault in return_argsort_stable tests."""

import jax
import jax.numpy as jnp
from tallax._src.test_utils import verify_sort_output
import sys

def test_case(name, **kwargs):
    """Test a specific case and report results."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Parameters: {kwargs}")
    print('='*60)

    shape = (16, 128)
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
        print(f"✓ PASSED: {name}")
        return True
    except Exception as e:
        print(f"✗ FAILED: {name}")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    results = {}

    # Test basic cases
    print("\n" + "="*60)
    print("PHASE 1: Basic functionality tests")
    print("="*60)

    results['standard'] = test_case(
        "Standard sort",
        return_argsort=False,
        is_stable=False,
        descending=False
    )

    results['return_argsort'] = test_case(
        "Return argsort",
        return_argsort=True,
        is_stable=False,
        descending=False
    )

    results['stable_no_argsort'] = test_case(
        "Stable sort (no argsort)",
        return_argsort=False,
        is_stable=True,
        descending=False
    )

    results['stable_with_argsort'] = test_case(
        "Stable sort WITH argsort",
        return_argsort=True,
        is_stable=True,
        descending=False
    )

    # Test descending variants
    print("\n" + "="*60)
    print("PHASE 2: Descending variants")
    print("="*60)

    results['descending'] = test_case(
        "Descending sort",
        return_argsort=False,
        is_stable=False,
        descending=True
    )

    results['descending_argsort'] = test_case(
        "Descending with argsort",
        return_argsort=True,
        is_stable=False,
        descending=True
    )

    results['descending_stable'] = test_case(
        "Descending stable (no argsort)",
        return_argsort=False,
        is_stable=True,
        descending=True
    )

    results['descending_stable_argsort'] = test_case(
        "Descending stable WITH argsort",
        return_argsort=True,
        is_stable=True,
        descending=True
    )

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    print("\nResults by test:")
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    # Identify pattern
    print("\n" + "="*60)
    print("PATTERN ANALYSIS")
    print("="*60)

    stable_argsort_tests = [
        'stable_with_argsort',
        'descending_stable_argsort'
    ]

    if all(not results.get(t, True) for t in stable_argsort_tests):
        print("❌ PATTERN: Segfault occurs when BOTH is_stable=True AND return_argsort=True")
    else:
        print("Pattern unclear from results")
