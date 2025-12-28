#!/usr/bin/env python3
"""Simple test for (8, 256) with and without compile_fast."""
import jax
import jax.numpy as jnp
from tallax import tax
from tallax._src.test_utils import verify_sort_output
from tallax._src.utils import is_cpu_platform

def test_sort(shape, compile_fast):
    """Test sort with given shape and compile_fast setting."""
    print(f"\n=== Testing shape {shape} with compile_fast={compile_fast} ===")

    dtype = jnp.float32
    num_arrays = 1
    num_keys = 1

    # Generate test data
    key = jax.random.key(42)
    operands = [jax.random.normal(key, shape, dtype=dtype)]

    interpret = is_cpu_platform()

    # Run sort
    if compile_fast is not None:
        outputs = tax.sort(
            operands,
            num_keys=num_keys,
            interpret=interpret,
            compile_fast=compile_fast
        )
    else:
        outputs = tax.sort(
            operands,
            num_keys=num_keys,
            interpret=interpret
        )

    # Verify outputs
    valid = verify_sort_output(
        operands,
        outputs,
        num_keys=num_keys,
        interpret=interpret
    )

    if valid:
        print(f"  ✓ Test PASSED")
        return True
    else:
        print(f"  ✗ Test FAILED")
        return False


if __name__ == "__main__":
    # Test without compile_fast (default)
    passed1 = test_sort((8, 256), compile_fast=None)

    # Test with compile_fast=False
    passed2 = test_sort((8, 256), compile_fast=False)

    # Test with compile_fast=True
    passed3 = test_sort((8, 256), compile_fast=True)

    if passed1 and passed2 and passed3:
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("✗ SOME TESTS FAILED")
        print("="*70)
