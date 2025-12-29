#!/usr/bin/env python
"""Test script to verify axis parameter restoration for specific shapes."""

try:
    import jax
    import jax.numpy as jnp
    from tallax import tax
    from tallax._src.test_utils import verify_sort_output
    from tallax._src.utils import is_cpu_platform

    def test_shape(batch_size, size, dtype=jnp.bfloat16):
        """Test sorting with specific shape and dtype."""
        print(f"\nTesting shape ({batch_size}, {size}) with dtype {dtype}")

        shape = (batch_size, size)
        key = jax.random.key(0)

        # Generate test data
        arr = jax.random.normal(key, shape, dtype=jnp.float32).astype(dtype)
        operands = [arr]

        # Use interpret mode on CPU
        interpret = is_cpu_platform()

        # Run sort (should use axis=1 by default)
        outputs = tax.sort(
            operands,
            num_keys=1,
            return_argsort=False,
            is_stable=False,
            descending=False,
            interpret=interpret
        )

        # Verify outputs
        valid = verify_sort_output(
            operands,
            outputs,
            num_keys=1,
            return_argsort=False,
            is_stable=False,
            descending=False,
            interpret=interpret
        )

        if valid:
            print(f"✓ Test PASSED for shape ({batch_size}, {size})")
        else:
            print(f"✗ Test FAILED for shape ({batch_size}, {size})")

        return valid

    # Run tests for the requested shapes
    print("=" * 60)
    print("Testing axis parameter restoration")
    print("=" * 60)

    results = []
    results.append(test_shape(137, 17, jnp.bfloat16))
    results.append(test_shape(259, 17, jnp.bfloat16))

    print("\n" + "=" * 60)
    if all(results):
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)

except ImportError as e:
    print(f"Cannot run tests: {e}")
    print("JAX or other dependencies are not installed.")
    print("The syntax validation passed, but runtime tests require JAX.")
