"""Debug test to understand what's happening in the kernel."""
import jax
import jax.numpy as jnp

# Test if the issue is in float_to_sortable_int when called inside Pallas
from jax.experimental import pallas as pl
from tallax._src.utils import float_to_sortable_int

def test_float_to_sortable_int_in_pallas():
    """Test if float_to_sortable_int works inside a Pallas kernel."""
    print("Testing float_to_sortable_int inside Pallas kernel...")

    def kernel(float_ref, out_ref):
        float_data = float_ref[...]
        int_data = float_to_sortable_int(float_data)
        out_ref[...] = int_data

    x = jnp.array([[3.0, 1.0, 4.0, 2.0]], dtype=jnp.float32)

    try:
        result = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, jnp.int32),
            interpret=True
        )(x)
        print(f"✓ SUCCESS: {result}")
        print(f"  Dtype: {result.dtype}\n")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_standardize_in_pallas():
    """Test if standardize (called by float_to_sortable_int) works in Pallas."""
    print("Testing standardize inside Pallas kernel...")

    from tallax._src.utils import standardize

    def kernel(float_ref, out_ref):
        float_data = float_ref[...]
        standardized = standardize(float_data)
        out_ref[...] = standardized

    x = jnp.array([[3.0, 1.0, jnp.nan, 2.0]], dtype=jnp.float32)

    try:
        result = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, jnp.float32),
            interpret=True
        )(x)
        print(f"✓ SUCCESS: {result}")
        print(f"  Has NaN: {jnp.isnan(result).any()}\n")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("Debug Tests for Pallas Kernel Operations")
    print("=" * 70)
    print()

    results = []
    results.append(("standardize in Pallas", test_standardize_in_pallas()))
    results.append(("float_to_sortable_int in Pallas", test_float_to_sortable_int_in_pallas()))

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
