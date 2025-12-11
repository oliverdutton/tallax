"""Test if Pallas interpret mode segfaults with view/bitcast operations."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def test_pallas_bitcast():
    """Test bitcast inside Pallas kernel with interpret mode."""
    print("Testing Pallas bitcast with interpret=True...")

    def kernel(in_ref, out_ref):
        x = in_ref[...]
        # Convert float32 to int32 view (bitcast)
        i = x.bitcast(jnp.int32)
        out_ref[...] = i

    x = jnp.array([[1.0, 2.0, 3.0, 4.0]], dtype=jnp.float32)

    try:
        result = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, jnp.int32),
            interpret=True
        )(x)
        print(f"✓ SUCCESS: bitcast worked in interpret mode")
        print(f"  Result: {result}")
    except Exception as e:
        print(f"✗ FAILED: {e}")

def test_pallas_view():
    """Test view inside Pallas kernel with interpret mode."""
    print("\nTesting Pallas view with interpret=True...")

    def kernel(in_ref, out_ref):
        x = in_ref[...]
        # Convert float32 to int32 view
        i = x.view(jnp.int32)
        out_ref[...] = i

    x = jnp.array([[1.0, 2.0, 3.0, 4.0]], dtype=jnp.float32)

    try:
        result = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, jnp.int32),
            interpret=True
        )(x)
        print(f"✓ SUCCESS: view worked in interpret mode")
        print(f"  Result: {result}")
    except Exception as e:
        print(f"✗ FAILED: {e}")

if __name__ == "__main__":
    test_pallas_bitcast()
    test_pallas_view()
