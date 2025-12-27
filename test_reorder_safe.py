#!/usr/bin/env python3
"""Test reordering in separate subprocesses to avoid full crash."""

import subprocess
import sys
import random

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

    tests = []
    for size in [128, 256]:
        for dtype_name in ['bfloat16', 'float32']:
            for variant_name, _ in variants:
                test_name = f"{dtype_name}-{size}-{variant_name}"
                tests.append(test_name)

    return tests


def run_sequence_script(test_names, sequence_name):
    """Generate and run a test script for a specific sequence."""
    # Create test script
    script = f'''
import jax
import jax.numpy as jnp
from tallax._src.test_utils import verify_sort_output

test_sequence = {test_names!r}

variants = {{
    "standard": {{"return_argsort": False, "is_stable": False, "descending": False}},
    "return_argsort": {{"return_argsort": True, "is_stable": False, "descending": False}},
    "stable": {{"return_argsort": False, "is_stable": True, "descending": False}},
    "stable_argsort": {{"return_argsort": True, "is_stable": True, "descending": False}},
    "descending": {{"return_argsort": False, "is_stable": False, "descending": True}},
    "descending_argsort": {{"return_argsort": True, "is_stable": False, "descending": True}},
    "descending_stable": {{"return_argsort": False, "is_stable": True, "descending": True}},
    "descending_stable_argsort": {{"return_argsort": True, "is_stable": True, "descending": True}},
}}

dtype_map = {{"bfloat16": jnp.bfloat16, "float32": jnp.float32}}

for i, test_name in enumerate(test_sequence, 1):
    parts = test_name.split("-")
    dtype_name = parts[0]
    size = int(parts[1])
    variant_name = "-".join(parts[2:])

    dtype = dtype_map[dtype_name]
    kwargs = variants[variant_name]

    print(f"[{{i:3d}}] {{test_name}}...", end=" ", flush=True)

    shape = (16, size)
    key = jax.random.key(0)
    arr = jax.random.normal(key, shape, dtype=jnp.float32).astype(dtype)
    operands = [arr]

    verify_sort_output(operands, num_keys=1, interpret=True, **kwargs)
    print("✓")

print("\\nSUCCESS: All {{len(test_sequence)}} tests passed")
'''

    print(f"\n{'='*70}")
    print(f"TESTING: {sequence_name}")
    print('='*70)

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300
    )

    # Parse output to find where it failed
    output_lines = result.stdout.split('\n')
    last_test_line = None
    for line in output_lines:
        if line.strip().startswith('['):
            last_test_line = line
            print(line)

    if result.returncode == -11:  # SIGSEGV
        # Extract last test number and name
        if last_test_line:
            parts = last_test_line.split(']')
            test_num = int(parts[0].strip('[').strip())
            test_name = parts[1].split('...')[0].strip()
            print(f"\n💥 SEGFAULT at position {test_num}: {test_name}")
            return test_num, test_name
        else:
            print("\n💥 SEGFAULT (couldn't determine position)")
            return None, None

    elif result.returncode == 0:
        if "SUCCESS" in result.stdout:
            print(f"\n✅ All {len(test_names)} tests passed!")
            return None, None
        else:
            print("\n⚠️  Completed but without success message")
            return None, None

    else:
        print(f"\n✗ Failed with exit code {result.returncode}")
        if result.stderr:
            print(f"Error: {result.stderr[-200:]}")
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
    pos1, test1 = run_sequence_script(base_tests, "Original order")
    results['original'] = (pos1, test1)

    # Test 2: Reversed order
    tests_reversed = list(reversed(base_tests))
    pos2, test2 = run_sequence_script(tests_reversed, "Reversed order")
    results['reversed'] = (pos2, test2)

    # Test 3: Move problematic test to position 1 (if we found one)
    if test1:
        tests_reordered = [test1] + [t for t in base_tests if t != test1]
        pos3, test3 = run_sequence_script(tests_reordered, f"'{test1}' moved to position 1")
        results['problematic_first'] = (pos3, test3)

    # Test 4: Random shuffle
    tests_shuffled = base_tests.copy()
    random.seed(42)
    random.shuffle(tests_shuffled)
    pos4, test4 = run_sequence_script(tests_shuffled, "Random shuffle (seed=42)")
    results['shuffled'] = (pos4, test4)

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    print("\nFailure positions:")
    for name, (pos, test) in results.items():
        if pos:
            print(f"  {name:20s}: Position {pos:2d} - {test}")
        else:
            print(f"  {name:20s}: ALL PASSED")

    positions = [r[0] for r in results.values() if r[0] is not None]
    failing_tests = [r[1] for r in results.values() if r[1] is not None]

    if positions:
        print("\n" + "="*70)
        print("CONCLUSION")
        print("="*70)

        # Check if same position fails
        if len(set(positions)) == 1:
            print(f"✓ SAME POSITION ({positions[0]}) fails in all runs")
            print("  → RESOURCE/ACCUMULATION ISSUE")
            print(f"  → After {positions[0]-1} successful compilations, test #{positions[0]} triggers crash")
            print("  → This is a JAX/XLA resource exhaustion bug")

        # Check if same test fails
        elif len(set(failing_tests)) == 1:
            print(f"✓ SAME TEST ({failing_tests[0]}) fails in all runs")
            print("  → TEST-CONTENT ISSUE")
            print("  → Something specific about this test configuration triggers the bug")

        elif len(set(positions)) <= 2:
            avg_pos = sum(positions) / len(positions)
            print(f"→ SIMILAR POSITIONS (avg: {avg_pos:.1f}, range: {min(positions)}-{max(positions)})")
            print("  → ACCUMULATION THRESHOLD")
            print(f"  → JAX/XLA crashes after approximately {int(avg_pos)} compilations")

        else:
            print("✗ Different positions AND different tests fail")
            print("  → COMPLEX INTERACTION or INTERMITTENT")
            avg_pos = sum(positions) / len(positions)
            print(f"  → Average failure position: {avg_pos:.1f}")
            print(f"  → Positions range: {min(positions)} to {max(positions)}")

    else:
        print("\n✅ NO FAILURES in any test sequence!")


if __name__ == "__main__":
    main()
