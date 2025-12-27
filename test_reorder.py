#!/usr/bin/env python3
"""Test if segfault is position-dependent or test-content-dependent."""

import jax
import jax.numpy as jnp
from tallax._src.test_utils import verify_sort_output
import sys
import random
import traceback

def create_test_sequence():
    """Create the full test sequence."""
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
    sizes = [128, 256]

    tests = []
    for size in sizes:
        for dtype in dtypes:
            for variant_name, kwargs in variants:
                test_name = f"{dtype.__name__}-{size}-{variant_name}"
                tests.append((test_name, size, dtype, kwargs))

    return tests


def run_test(test_num, test_name, size, dtype, kwargs):
    """Run a single test."""
    shape = (16, size)

    print(f"[{test_num:3d}] {test_name}...", end=" ", flush=True)

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
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {type(e).__name__}")
        return False


def run_sequence(tests, sequence_name):
    """Run a sequence of tests and report where it fails."""
    print("\n" + "="*70)
    print(f"SEQUENCE: {sequence_name}")
    print("="*70)

    for i, (test_name, size, dtype, kwargs) in enumerate(tests, 1):
        success = run_test(i, test_name, size, dtype, kwargs)
        if not success:
            print(f"\n💥 SEGFAULT/FAILURE at position {i}: {test_name}")
            print(f"    This was test: {test_name}")
            return i, test_name

    print(f"\n✅ All {len(tests)} tests passed!")
    return None, None


def main():
    """Run tests in different orders."""
    base_tests = create_test_sequence()

    print("="*70)
    print("REORDERING TEST TO ISOLATE SEGFAULT CAUSE")
    print("="*70)
    print(f"Total tests: {len(base_tests)}")
    print()
    print("If same position fails → resource/accumulation issue")
    print("If same test fails → test-content issue")
    print("="*70)

    results = {}

    # Test 1: Original order
    print("\n" + "="*70)
    print("TEST 1: ORIGINAL ORDER")
    print("="*70)
    tests_original = base_tests.copy()
    pos1, test1 = run_sequence(tests_original, "Original order")
    results['original'] = (pos1, test1)

    # Test 2: Reversed order
    print("\n" + "="*70)
    print("TEST 2: REVERSED ORDER")
    print("="*70)
    tests_reversed = list(reversed(base_tests))
    pos2, test2 = run_sequence(tests_reversed, "Reversed order")
    results['reversed'] = (pos2, test2)

    # Test 3: Move the problematic test to position 1
    if test1:
        print("\n" + "="*70)
        print(f"TEST 3: PROBLEMATIC TEST ({test1}) MOVED TO POSITION 1")
        print("="*70)
        tests_reordered = [t for t in base_tests if t[0] == test1]
        tests_reordered += [t for t in base_tests if t[0] != test1]
        pos3, test3 = run_sequence(tests_reordered, "Problematic test first")
        results['problematic_first'] = (pos3, test3)

    # Test 4: Random shuffle
    print("\n" + "="*70)
    print("TEST 4: RANDOM SHUFFLE")
    print("="*70)
    tests_shuffled = base_tests.copy()
    random.seed(42)
    random.shuffle(tests_shuffled)
    pos4, test4 = run_sequence(tests_shuffled, "Random shuffle")
    results['shuffled'] = (pos4, test4)

    # Summary
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    positions = [r[0] for r in results.values() if r[0] is not None]
    failing_tests = [r[1] for r in results.values() if r[1] is not None]

    print("\nFailure positions:")
    for name, (pos, test) in results.items():
        if pos:
            print(f"  {name:20s}: Position {pos:2d} - {test}")
        else:
            print(f"  {name:20s}: ALL PASSED")

    if positions:
        print("\n" + "="*70)
        print("CONCLUSION")
        print("="*70)

        # Check if same position fails
        if len(set(positions)) == 1:
            print(f"✓ Same position ({positions[0]}) fails in all runs")
            print("  → RESOURCE/ACCUMULATION ISSUE")
            print(f"  → After {positions[0]-1} successful compilations, the {positions[0]}th fails")
            print("  → Likely JAX/XLA memory leak or resource exhaustion")

        # Check if same test fails
        elif len(set(failing_tests)) == 1:
            print(f"✓ Same test ({failing_tests[0]}) fails in all runs")
            print("  → TEST-CONTENT ISSUE")
            print("  → Something specific about this test triggers the bug")

        else:
            print("✗ Different positions AND different tests fail")
            print("  → COMPLEX INTERACTION")
            print("  → May be combination of test order and content")

            # Check for patterns
            avg_pos = sum(positions) / len(positions)
            print(f"\n  Average failure position: {avg_pos:.1f}")
            print(f"  Positions range: {min(positions)} to {max(positions)}")
    else:
        print("\n✅ NO FAILURES in any test sequence!")
        print("  → Segfault may be intermittent or environment-specific")


if __name__ == "__main__":
    main()
