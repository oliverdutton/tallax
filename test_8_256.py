#!/usr/bin/env python3
"""Test sort with (8, 256) shape and compile_fast=True."""
import jax
import jax.numpy as jnp
from tallax import tax
from tallax._src.test_utils import verify_sort_output
from tallax._src.utils import is_cpu_platform

def test_8_256_compile_fast():
    """Test shape (8, 256) with compile_fast=True."""
    print("\n=== Testing shape (8, 256) with compile_fast=True ===")

    shape = (8, 256)
    dtype = jnp.float32
    num_arrays = 1
    num_keys = 1

    # Test a few key variants
    test_variants = [
        ("default", False, False, False),
        ("return_argsort", True, False, False),
        ("stable", False, True, False),
        ("descending", False, False, True),
    ]

    for variant_name, return_argsort, is_stable, descending in test_variants:
        print(f"\nTesting variant: {variant_name} with compile_fast=True")

        # Generate test data
        key = jax.random.key(42)
        operands = [jax.random.normal(key, shape, dtype=dtype)]

        interpret = is_cpu_platform()

        # Run sort with compile_fast=True
        outputs = tax.sort(
            operands,
            num_keys=num_keys,
            return_argsort=return_argsort,
            is_stable=is_stable,
            descending=descending,
            interpret=interpret,
            compile_fast=True  # Set compile_fast=True
        )

        # Verify outputs
        valid = verify_sort_output(
            operands,
            outputs,
            num_keys=num_keys,
            return_argsort=return_argsort,
            is_stable=is_stable,
            descending=descending,
            interpret=interpret
        )

        if valid:
            print(f"  ✓ {variant_name} passed with compile_fast=True")
        else:
            print(f"  ✗ {variant_name} FAILED with compile_fast=True")
            raise AssertionError(f"Sort validation failed for {variant_name} with compile_fast=True")

    print("\n✓ All variants passed for (8, 256) with compile_fast=True!")


if __name__ == "__main__":
    test_8_256_compile_fast()
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED!")
    print("="*70)
