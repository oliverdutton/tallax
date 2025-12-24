"""Simplified test for bitonic_sort CSE - just test jaxpr extraction and CSE."""

import jax
import jax.numpy as jnp
from jax import make_jaxpr
from jax.experimental import pallas as pl
from tallax._src.bitonic_sort import bitonic_sort
from tallax._src.cse import cse_until_fixpoint

def test_cse_simple():
    """Test CSE on bitonic_sort jaxpr extraction."""

    shape = (8, 1024)
    print(f"\n{'='*80}")
    print(f"Testing CSE on bitonic_sort jaxpr (shape {shape})")
    print(f"{'='*80}\n")

    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, shape, dtype=jnp.float32)

    print(f"Input shape: {x.shape}")

    # Test with CSE enabled
    print(f"\n{'-'*80}")
    print("Running bitonic_sort with apply_cse=True")
    print(f"{'-'*80}\n")

    try:
        result = bitonic_sort(x, descending=False, apply_cse=True, interpret=True)
        result = result[0]

        print(f"\nOutput shape: {result.shape}")
        print(f"First few values: {result[0, :10]}")

        # Check if sorted
        is_sorted = jnp.all(result[:, :-1] <= result[:, 1:])
        print(f"Is sorted: {is_sorted}")

        if is_sorted:
            print("\n✓ SUCCESS: CSE'd bitonic_sort produces sorted output!")
            return True
        else:
            print("\n✗ FAILURE: Output is not sorted")
            return False

    except Exception as e:
        print(f"\n✗ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_cse_simple()
    exit(0 if success else 1)
