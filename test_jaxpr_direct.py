"""Test extracting jaxpr and running directly without CSE."""

import jax
import jax.numpy as jnp
from jax import make_jaxpr
from jax.extend.core import jaxpr_as_fun
from tallax._src.bitonic_sort import bitonic_sort_arrays

def test_jaxpr_direct(shape):
    """Test extracting jaxpr from bitonic_sort_arrays and running directly."""

    print(f"\n{'='*80}")
    print(f"Testing jaxpr extraction and direct execution on shape {shape}")
    print(f"{'='*80}\n")

    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, shape, dtype=jnp.float32)

    print(f"Input shape: {x.shape}")

    # Test 1: Direct execution
    print(f"\n{'-'*80}")
    print("Test 1: Direct execution of bitonic_sort_arrays")
    print(f"{'-'*80}\n")

    result_direct = bitonic_sort_arrays(
        [x],
        num_keys=1,
        descending=False,
        unroll_stages=True,
    )[0]

    print(f"Output shape: {result_direct.shape}")
    print(f"First row sample: {result_direct[0, :10]}")
    is_sorted_direct = jnp.all(result_direct[:, :-1] <= result_direct[:, 1:])
    print(f"Is sorted: {is_sorted_direct}")

    # Test 2: Via jaxpr
    print(f"\n{'-'*80}")
    print("Test 2: Via jaxpr_as_fun (no CSE)")
    print(f"{'-'*80}\n")

    def sort_fn(arr):
        return bitonic_sort_arrays(
            [arr],
            num_keys=1,
            descending=False,
            unroll_stages=True,
        )

    print("Extracting jaxpr...")
    closed_jaxpr = make_jaxpr(sort_fn)(x)
    print(f"Jaxpr has {len(closed_jaxpr.jaxpr.eqns)} equations")

    print("Running via jaxpr_as_fun...")
    jaxpr_fn = jaxpr_as_fun(closed_jaxpr)
    result_jaxpr = jaxpr_fn(x)[0]

    print(f"Output shape: {result_jaxpr.shape}")
    print(f"First row sample: {result_jaxpr[0, :10]}")
    is_sorted_jaxpr = jnp.all(result_jaxpr[:, :-1] <= result_jaxpr[:, 1:])
    print(f"Is sorted: {is_sorted_jaxpr}")

    # Compare
    print(f"\n{'-'*80}")
    print("Comparison")
    print(f"{'-'*80}\n")

    match = jnp.allclose(result_direct, result_jaxpr)
    print(f"Results match: {match}")

    if is_sorted_direct and is_sorted_jaxpr and match:
        print(f"\n✓ SUCCESS: jaxpr execution works correctly!")
        return True
    else:
        print(f"\n✗ FAILURE: Something is wrong")
        return False


if __name__ == "__main__":
    success = test_jaxpr_direct((16, 256))
    exit(0 if success else 1)
