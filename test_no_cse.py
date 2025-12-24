"""Test bitonic_sort WITHOUT CSE to verify it works."""

import jax
import jax.numpy as jnp
from tallax._src.bitonic_sort import bitonic_sort

def test_no_cse(shape):
    """Test bitonic_sort without CSE on given shape."""

    print(f"\n{'='*80}")
    print(f"Testing bitonic_sort WITHOUT CSE on shape {shape}")
    print(f"{'='*80}\n")

    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, shape, dtype=jnp.float32)

    print(f"Input shape: {x.shape}")

    # Test without CSE
    print(f"\n{'-'*80}")
    print("Running with apply_cse=False (direct execution)")
    print(f"{'-'*80}\n")

    result = bitonic_sort(x, descending=False, apply_cse=False, interpret=True)
    result = result[0]

    print(f"\nOutput shape: {result.shape}")
    print(f"First row sample: {result[0, :10]}")

    # Check if sorted
    is_sorted = jnp.all(result[:, :-1] <= result[:, 1:])
    print(f"Is sorted: {is_sorted}")

    if is_sorted:
        print(f"\n✓ SUCCESS: Direct bitonic_sort works correctly!")
        return True
    else:
        print(f"\n✗ FAILURE: Output is not sorted")
        return False


if __name__ == "__main__":
    success = test_no_cse((16, 256))
    exit(0 if success else 1)
