#!/usr/bin/env python3
"""Test (16, 256) shape with all kwarg variants to identify segfault conditions."""

import jax
import jax.numpy as jnp
from tallax._src.test_utils import verify_sort_output
import sys
import subprocess
import os

def test_single_variant(name, **kwargs):
    """Test a single variant in isolation."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Parameters: {kwargs}")
    print('='*60)

    shape = (16, 256)
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


def run_individual_tests():
    """Run each test variant in a separate subprocess."""
    test_variants = [
        ("standard", {"return_argsort": False, "is_stable": False, "descending": False}),
        ("return_argsort", {"return_argsort": True, "is_stable": False, "descending": False}),
        ("stable", {"return_argsort": False, "is_stable": True, "descending": False}),
        ("stable_argsort", {"return_argsort": True, "is_stable": True, "descending": False}),
        ("descending", {"return_argsort": False, "is_stable": False, "descending": True}),
        ("descending_argsort", {"return_argsort": True, "is_stable": False, "descending": True}),
        ("descending_stable", {"return_argsort": False, "is_stable": True, "descending": True}),
        ("descending_stable_argsort", {"return_argsort": True, "is_stable": True, "descending": True}),
    ]

    print("\n" + "="*60)
    print("PHASE 1: Running each variant in separate subprocess")
    print("="*60)

    results = {}
    for name, kwargs in test_variants:
        # Create a minimal test script
        script = f"""
import jax
import jax.numpy as jnp
from tallax._src.test_utils import verify_sort_output

shape = (16, 256)
key = jax.random.key(0)
arr = jax.random.normal(key, shape, dtype=jnp.float32).astype(jnp.bfloat16)
operands = [arr]

verify_sort_output(
    operands,
    num_keys=1,
    interpret=True,
    {', '.join(f'{k}={v}' for k, v in kwargs.items())}
)
print("SUCCESS")
"""

        print(f"\nRunning {name} in subprocess...")
        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=os.getcwd()
            )

            if result.returncode == 0 and "SUCCESS" in result.stdout:
                print(f"✓ PASSED: {name}")
                results[name] = True
            elif result.returncode == -11:  # SIGSEGV
                print(f"✗ SEGFAULT: {name}")
                results[name] = "SEGFAULT"
            else:
                print(f"✗ FAILED: {name} (exit code: {result.returncode})")
                if result.stderr:
                    print(f"Error output: {result.stderr[-500:]}")
                results[name] = False
        except subprocess.TimeoutExpired:
            print(f"✗ TIMEOUT: {name}")
            results[name] = "TIMEOUT"
        except Exception as e:
            print(f"✗ ERROR: {name}: {e}")
            results[name] = False

    return results


def run_all_in_sequence():
    """Run all tests in sequence in the same process."""
    print("\n" + "="*60)
    print("PHASE 2: Running all variants in sequence (same process)")
    print("="*60)

    test_variants = [
        ("standard", {"return_argsort": False, "is_stable": False, "descending": False}),
        ("return_argsort", {"return_argsort": True, "is_stable": False, "descending": False}),
        ("stable", {"return_argsort": False, "is_stable": True, "descending": False}),
        ("stable_argsort", {"return_argsort": True, "is_stable": True, "descending": False}),
        ("descending", {"return_argsort": False, "is_stable": False, "descending": True}),
        ("descending_argsort", {"return_argsort": True, "is_stable": False, "descending": True}),
        ("descending_stable", {"return_argsort": False, "is_stable": True, "descending": True}),
        ("descending_stable_argsort", {"return_argsort": True, "is_stable": True, "descending": True}),
    ]

    results = {}
    for name, kwargs in test_variants:
        results[name] = test_single_variant(name, **kwargs)

    return results


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode in ["individual", "both"]:
        print("\n" + "="*70)
        print("TESTING EACH VARIANT INDIVIDUALLY (separate subprocesses)")
        print("="*70)
        individual_results = run_individual_tests()

        print("\n" + "="*60)
        print("INDIVIDUAL TEST RESULTS")
        print("="*60)
        for name, result in individual_results.items():
            if result is True:
                status = "✓ PASS"
            elif result == "SEGFAULT":
                status = "💥 SEGFAULT"
            elif result == "TIMEOUT":
                status = "⏱ TIMEOUT"
            else:
                status = "✗ FAIL"
            print(f"  {status}: {name}")

    if mode in ["sequence", "both"]:
        print("\n" + "="*70)
        print("TESTING ALL VARIANTS IN SEQUENCE (same process)")
        print("="*70)
        sequence_results = run_all_in_sequence()

        print("\n" + "="*60)
        print("SEQUENTIAL TEST RESULTS")
        print("="*60)
        passed = sum(1 for v in sequence_results.values() if v)
        total = len(sequence_results)
        print(f"Passed: {passed}/{total}")
        for name, result in sequence_results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {status}: {name}")

    if mode == "both":
        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        print("If segfaults only happen in sequence, it suggests:")
        print("  - JAX/XLA state accumulation issue")
        print("  - Memory leak or corruption")
        print("If segfaults happen in individual runs too:")
        print("  - Specific kwarg combinations trigger the bug")
        print("  - JAX compilation issue for certain configurations")
