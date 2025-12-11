"""Minimal test to reproduce sort segfault on CPU."""
import jax
import jax.numpy as jnp
from tallax._src.test_utils import verify_sort_output
from tallax._src.utils import is_cpu_platform

def test_segfault_conditions():
    """Test various conditions to characterize the segfault."""

    print(f"Platform: {'CPU' if is_cpu_platform() else 'Other'}")
    print(f"JAX version: {jax.__version__}")

    shape = (16, 128)
    key = jax.random.key(0)

    # Test configurations
    configs = [
        ("bfloat16, return_argsort=True", jnp.bfloat16, True),
        ("float32, return_argsort=True", jnp.float32, True),
        ("float32, return_argsort=False", jnp.float32, False),
    ]

    for name, dtype, return_argsort in configs:
        print(f"\nTesting {name}...")
        try:
            if dtype == jnp.bfloat16:
                arr = jax.random.normal(key, shape, dtype=jnp.float32).astype(jnp.bfloat16)
            else:
                arr = jax.random.normal(key, shape, dtype=dtype)

            operands = [arr]
            verify_sort_output(
                operands,
                num_keys=1,
                return_argsort=return_argsort,
                is_stable=False,
                descending=False,
                interpret=is_cpu_platform()
            )
            print(f"✓ PASSED: {name}")
        except Exception as e:
            print(f"✗ FAILED: {name}")
            print(f"  Error: {e}")

if __name__ == "__main__":
    test_segfault_conditions()
