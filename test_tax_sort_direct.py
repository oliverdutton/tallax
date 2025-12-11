"""Test calling tax.sort directly to isolate the segfault."""
import jax
import jax.numpy as jnp
from tallax import tax

def test_tax_sort_no_argsort():
    """Test tax.sort without return_argsort."""
    print("Testing tax.sort without return_argsort (float32)...")
    key = jax.random.key(0)
    x = jax.random.normal(key, (16, 128), dtype=jnp.float32)
    try:
        result = tax.sort([x], num_keys=1, return_argsort=False, interpret=True)
        print(f"✓ SUCCESS: result shape = {result[0].shape}, dtype = {result[0].dtype}")
    except Exception as e:
        print(f"✗ FAILED: {e}")

def test_tax_sort_with_argsort():
    """Test tax.sort WITH return_argsort."""
    print("\nTesting tax.sort WITH return_argsort (float32)...")
    print("WARNING: This is expected to segfault...")
    key = jax.random.key(0)
    x = jax.random.normal(key, (16, 128), dtype=jnp.float32)
    try:
        result = tax.sort([x], num_keys=1, return_argsort=True, interpret=True)
        print(f"✓ SUCCESS: result shapes = {[r.shape for r in result]}")
    except Exception as e:
        print(f"✗ FAILED: {e}")

def test_tax_sort_bfloat16_with_argsort():
    """Test tax.sort with bfloat16 and return_argsort."""
    print("\nTesting tax.sort WITH return_argsort (bfloat16)...")
    key = jax.random.key(0)
    x = jax.random.normal(key, (16, 128), dtype=jnp.float32).astype(jnp.bfloat16)
    try:
        result = tax.sort([x], num_keys=1, return_argsort=True, interpret=True)
        print(f"✓ SUCCESS: result shapes = {[r.shape for r in result]}, dtypes = {[r.dtype for r in result]}")
    except Exception as e:
        print(f"✗ FAILED: {e}")

if __name__ == "__main__":
    test_tax_sort_no_argsort()
    test_tax_sort_bfloat16_with_argsort()
    # Commenting out the segfaulting test
    # test_tax_sort_with_argsort()
