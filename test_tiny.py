"""Test with very small size."""

import jax
import jax.numpy as jnp
from tallax._src.bitonic_sort import bitonic_sort_arrays

def test_tiny():
    """Test with tiny size."""

    shape = (8, 128)
    print(f"\nTesting with shape {shape}")

    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, shape, dtype=jnp.float32)

    print(f"Input: {x[0, :10]}")

    result = bitonic_sort_arrays(
        [x],
        num_keys=1,
        descending=False,
        unroll_stages=True,
    )[0]

    print(f"Output: {result[0, :10]}")

    is_sorted = jnp.all(result[:, :-1] <= result[:, 1:])
    print(f"Is sorted: {is_sorted}")

    if jnp.any(jnp.isnan(result)):
        print("✗ Contains NaN")
        return False
    elif is_sorted:
        print("✓ Correctly sorted")
        return True
    else:
        print("✗ Not sorted correctly")
        return False


if __name__ == "__main__":
    success = test_tiny()
    exit(0 if success else 1)
