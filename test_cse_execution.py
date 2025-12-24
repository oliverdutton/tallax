"""Test bitonic_sort with CSE - smaller test first."""

import jax
import jax.numpy as jnp
from tallax._src.bitonic_sort import bitonic_sort

def test_cse_execution(shape):
    """Test bitonic_sort with CSE on given shape."""

    print(f"\n{'='*80}")
    print(f"Testing bitonic_sort with CSE on shape {shape}")
    print(f"{'='*80}\n")

    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, shape, dtype=jnp.float32)

    print(f"Input shape: {x.shape}")

    # Test with CSE
    print(f"\n{'-'*80}")
    print("Running with apply_cse=True")
    print(f"{'-'*80}\n")

    result = bitonic_sort(x, descending=False, apply_cse=True)
    result = result[0]

    print(f"\nOutput shape: {result.shape}")
    print(f"First row sample: {result[0, :10]}")

    # Check if sorted
    is_sorted = jnp.all(result[:, :-1] <= result[:, 1:])
    print(f"Is sorted: {is_sorted}")

    if is_sorted:
        print(f"\n✓ SUCCESS: CSE'd bitonic_sort produces correctly sorted output!")
        return True
    else:
        print(f"\n✗ FAILURE: Output is not sorted")
        return False


if __name__ == "__main__":
    # Test (16, 256) first
    print("Testing smaller size first...")
    success1 = test_cse_execution((16, 256))

    if success1:
        # If that works, try (8, 1024)
        print("\n\nSmaller test passed, trying (8, 1024)...")
        success2 = test_cse_execution((8, 1024))
        exit(0 if success2 else 1)
    else:
        exit(1)
