"""Test to understand dtype handling in sort."""
import jax
import jax.numpy as jnp
from tallax._src.utils import is_cpu_platform, float_to_sortable_int
from tallax import tax

def test_conversion():
    """Check what dtypes are used at each stage."""
    shape = (16, 128)
    key = jax.random.key(0)

    # Test with float32
    arr_f32 = jax.random.normal(key, shape, dtype=jnp.float32)
    print(f"Original float32 array dtype: {arr_f32.dtype}")

    # Convert to sortable int
    arr_int = float_to_sortable_int(arr_f32)
    print(f"After float_to_sortable_int: {arr_int.dtype}")

    # Try calling sort without return_argsort (should work)
    print("\n--- Testing without return_argsort ---")
    try:
        result = tax.sort([arr_f32], num_keys=1, return_argsort=False, interpret=True)
        print(f"✓ SUCCESS: sort without argsort")
        print(f"  Result dtype: {result[0].dtype}")
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Try calling sort with return_argsort (expected to segfault)
    print("\n--- Testing with return_argsort ---")
    print("WARNING: This may segfault...")
    # Don't actually run this to avoid hanging

if __name__ == "__main__":
    test_conversion()
