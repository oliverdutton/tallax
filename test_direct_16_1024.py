"""Test direct call to bitonic_sort_arrays with (16, 1024)."""

import jax
import jax.numpy as jnp
from tallax._src.bitonic_sort import bitonic_sort_arrays

def test_direct_call():
    """Test bitonic_sort_arrays directly with (16, 1024)."""

    shape = (16, 1024)
    print(f"\n{'='*80}")
    print(f"Direct call to bitonic_sort_arrays with shape {shape}")
    print(f"{'='*80}\n")

    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, shape, dtype=jnp.float32)

    print(f"Input shape: {x.shape}")
    print(f"Input dtype: {x.dtype}")
    print(f"Sample input values (first row): {x[0, :10]}")

    print(f"\nCalling bitonic_sort_arrays with unroll_stages=True...")
    result = bitonic_sort_arrays(
        [x],
        num_keys=1,
        descending=False,
        unroll_stages=True,
    )[0]

    print(f"\nOutput shape: {result.shape}")
    print(f"Output dtype: {result.dtype}")
    print(f"Sample output values (first row): {result[0, :10]}")

    # Check for NaNs
    has_nan = jnp.any(jnp.isnan(result))
    print(f"\nContains NaN: {has_nan}")

    # Check if sorted
    is_sorted = jnp.all(result[:, :-1] <= result[:, 1:])
    print(f"Is sorted: {is_sorted}")

    # Check each row
    print(f"\nChecking individual rows:")
    for i in range(min(5, shape[0])):
        row_sorted = jnp.all(result[i, :-1] <= result[i, 1:])
        print(f"  Row {i}: sorted={row_sorted}, min={jnp.min(result[i]):.4f}, max={jnp.max(result[i]):.4f}")

    if has_nan:
        print("\n✗ FAILURE: Output contains NaN values")
        return False
    elif is_sorted:
        print("\n✓ SUCCESS: Output is correctly sorted!")
        return True
    else:
        print("\n✗ FAILURE: Output is not sorted")
        return False


if __name__ == "__main__":
    success = test_direct_call()
    exit(0 if success else 1)
