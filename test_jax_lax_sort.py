"""Test JAX lax.sort directly to see if it causes the segfault."""
import jax
import jax.numpy as jnp

def test_lax_sort_single_array():
    """Test lax.sort with a single array."""
    print("Testing lax.sort with single float32 array...")
    x = jnp.array([[3.0, 1.0, 4.0, 2.0]], dtype=jnp.float32)
    try:
        result = jax.lax.sort(x, dimension=1)
        print(f"✓ SUCCESS: {result}")
    except Exception as e:
        print(f"✗ FAILED: {e}")

def test_lax_sort_with_indices():
    """Test lax.sort with array + indices (like return_argsort)."""
    print("\nTesting lax.sort with float32 array + indices...")
    x = jnp.array([[3.0, 1.0, 4.0, 2.0]], dtype=jnp.float32)
    indices = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
    try:
        result = jax.lax.sort([x, indices], num_keys=1)
        print(f"✓ SUCCESS: {result}")
    except Exception as e:
        print(f"✗ FAILED: {e}")

def test_lax_sort_larger():
    """Test lax.sort with larger array."""
    print("\nTesting lax.sort with larger float32 array + indices...")
    key = jax.random.key(0)
    x = jax.random.normal(key, (16, 128), dtype=jnp.float32)
    indices = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    print(f"  Array shape: {x.shape}, dtype: {x.dtype}")
    print(f"  Indices shape: {indices.shape}, dtype: {indices.dtype}")
    try:
        result = jax.lax.sort([x, indices], num_keys=1)
        print(f"✓ SUCCESS: result shapes = {[r.shape for r in result]}")
    except Exception as e:
        print(f"✗ FAILED: {e}")

if __name__ == "__main__":
    test_lax_sort_single_array()
    test_lax_sort_with_indices()
    test_lax_sort_larger()
