"""Exactly mirror the test setup to understand the difference."""
import jax
import jax.numpy as jnp
from tallax._src.test_utils import verify_sort_output
from tallax._src.utils import is_cpu_platform

def test_bfloat16_exact():
    """Exact copy of test setup for bfloat16."""
    dtype = jnp.bfloat16
    size = 128
    shape = (16, size)
    key = jax.random.key(0)

    # Generate operands exactly as in test
    keys = jax.random.split(key, 1)
    operands = []
    for i in range(1):
        if dtype == jnp.bfloat16:
            arr = jax.random.normal(keys[i], shape, dtype=jnp.float32).astype(jnp.bfloat16)
        else:
            arr = jax.random.normal(keys[i], shape, dtype=dtype)
        operands.append(arr)

    print(f"Testing bfloat16 with return_argsort...")
    print(f"  Shape: {shape}, dtype: {dtype}")
    print(f"  Interpret: {is_cpu_platform()}")

    try:
        verify_sort_output(
            operands,
            num_keys=1,
            return_argsort=True,
            is_stable=False,
            descending=False,
            interpret=is_cpu_platform()
        )
        print("✓ PASSED")
    except Exception as e:
        print(f"✗ FAILED: {e}")

def test_float32_exact():
    """Exact copy of test setup for float32."""
    dtype = jnp.float32
    size = 128
    shape = (16, size)
    key = jax.random.key(0)

    # Generate operands exactly as in test
    keys = jax.random.split(key, 1)
    operands = []
    for i in range(1):
        arr = jax.random.normal(keys[i], shape, dtype=dtype)
        operands.append(arr)

    print(f"\nTesting float32 with return_argsort...")
    print(f"  Shape: {shape}, dtype: {dtype}")
    print(f"  Interpret: {is_cpu_platform()}")
    print("  WARNING: Expected to segfault!")

    # Don't actually run this to avoid hanging

if __name__ == "__main__":
    test_bfloat16_exact()
    # test_float32_exact()  # Commented out to avoid segfault
