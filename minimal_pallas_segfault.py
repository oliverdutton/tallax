"""Minimal Pallas code to reproduce the segfault without using tax.sort."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def test_single_array_sort():
    """Test Pallas with single array - should work."""
    print("Test 1: Single array sort in Pallas interpret mode")

    def kernel(x_ref, out_ref):
        x = x_ref[...]
        # Simple operation on the array
        out_ref[...] = jnp.sort(x, axis=-1)

    x = jnp.array([[3.0, 1.0, 4.0, 2.0]], dtype=jnp.float32)

    try:
        result = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            interpret=True
        )(x)
        print(f"✓ SUCCESS: {result}\n")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        return False

def test_two_array_passthrough():
    """Test Pallas with two arrays, just pass through - should work."""
    print("Test 2: Two arrays passthrough in Pallas interpret mode")

    def kernel(x_ref, idx_ref, out_x_ref, out_idx_ref):
        out_x_ref[...] = x_ref[...]
        out_idx_ref[...] = idx_ref[...]

    x = jnp.array([[3.0, 1.0, 4.0, 2.0]], dtype=jnp.float32)
    idx = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)

    try:
        result = pl.pallas_call(
            kernel,
            out_shape=[
                jax.ShapeDtypeStruct(x.shape, x.dtype),
                jax.ShapeDtypeStruct(idx.shape, idx.dtype)
            ],
            interpret=True
        )(x, idx)
        print(f"✓ SUCCESS: x={result[0]}, idx={result[1]}\n")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        return False

def test_two_array_sort_separate():
    """Test Pallas with two arrays, sort each separately - might work."""
    print("Test 3: Two arrays, sort each separately in Pallas interpret mode")

    def kernel(x_ref, idx_ref, out_x_ref, out_idx_ref):
        x = x_ref[...]
        idx = idx_ref[...]
        out_x_ref[...] = jnp.sort(x, axis=-1)
        out_idx_ref[...] = jnp.sort(idx, axis=-1)

    x = jnp.array([[3.0, 1.0, 4.0, 2.0]], dtype=jnp.float32)
    idx = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)

    try:
        result = pl.pallas_call(
            kernel,
            out_shape=[
                jax.ShapeDtypeStruct(x.shape, x.dtype),
                jax.ShapeDtypeStruct(idx.shape, idx.dtype)
            ],
            interpret=True
        )(x, idx)
        print(f"✓ SUCCESS: x={result[0]}, idx={result[1]}\n")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        return False

def test_two_array_argsort():
    """Test Pallas with argsort on one array - might work."""
    print("Test 4: Argsort in Pallas interpret mode")

    def kernel(x_ref, out_x_ref, out_idx_ref):
        x = x_ref[...]
        perm = jnp.argsort(x, axis=-1)
        out_idx_ref[...] = perm
        out_x_ref[...] = jnp.take_along_axis(x, perm, axis=-1)

    x = jnp.array([[3.0, 1.0, 4.0, 2.0]], dtype=jnp.float32)

    try:
        result = pl.pallas_call(
            kernel,
            out_shape=[
                jax.ShapeDtypeStruct(x.shape, x.dtype),
                jax.ShapeDtypeStruct(x.shape, jnp.int32)
            ],
            interpret=True
        )(x)
        print(f"✓ SUCCESS: x={result[0]}, idx={result[1]}\n")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        return False

def test_lax_sort_two_arrays():
    """Test lax.sort with two arrays inside Pallas - this might segfault!"""
    print("Test 5: lax.sort with two arrays in Pallas interpret mode")
    print("WARNING: This is expected to segfault if it's the root cause...")

    def kernel(x_ref, idx_ref, out_x_ref, out_idx_ref):
        x = x_ref[...]
        idx = idx_ref[...]
        # Sort two arrays together using lax.sort
        sorted_x, sorted_idx = jax.lax.sort([x, idx], num_keys=1)
        out_x_ref[...] = sorted_x
        out_idx_ref[...] = sorted_idx

    x = jnp.array([[3.0, 1.0, 4.0, 2.0]], dtype=jnp.float32)
    idx = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)

    try:
        result = pl.pallas_call(
            kernel,
            out_shape=[
                jax.ShapeDtypeStruct(x.shape, x.dtype),
                jax.ShapeDtypeStruct(idx.shape, idx.dtype)
            ],
            interpret=True
        )(x, idx)
        print(f"✓ SUCCESS: x={result[0]}, idx={result[1]}\n")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        return False

def test_lax_sort_two_arrays_larger():
    """Test lax.sort with larger arrays (128 elements) - closer to real test."""
    print("Test 6: lax.sort with two larger arrays (16x128) in Pallas interpret mode")
    print("WARNING: This is expected to segfault...")

    def kernel(x_ref, idx_ref, out_x_ref, out_idx_ref):
        x = x_ref[...]
        idx = idx_ref[...]
        # Sort two arrays together using lax.sort
        sorted_x, sorted_idx = jax.lax.sort([x, idx], num_keys=1)
        out_x_ref[...] = sorted_x
        out_idx_ref[...] = sorted_idx

    key = jax.random.key(0)
    x = jax.random.normal(key, (16, 128), dtype=jnp.float32)
    idx = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)

    try:
        result = pl.pallas_call(
            kernel,
            out_shape=[
                jax.ShapeDtypeStruct(x.shape, x.dtype),
                jax.ShapeDtypeStruct(idx.shape, idx.dtype)
            ],
            interpret=True
        )(x, idx)
        print(f"✓ SUCCESS: x shape={result[0].shape}, idx shape={result[1].shape}\n")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("Minimal Pallas Segfault Investigation")
    print("=" * 70)
    print()

    # Run tests in order of increasing complexity
    results = []

    results.append(("Single array sort", test_single_array_sort()))
    results.append(("Two arrays passthrough", test_two_array_passthrough()))
    results.append(("Two arrays sort separately", test_two_array_sort_separate()))
    results.append(("Argsort", test_two_array_argsort()))

    # These might segfault, so comment them out by default
    print("=" * 70)
    print("POTENTIALLY DANGEROUS TESTS (may segfault)")
    print("Uncomment in the code to run")
    print("=" * 70)
    print()

    # Uncomment to test:
    results.append(("lax.sort two arrays (small)", test_lax_sort_two_arrays()))
    results.append(("lax.sort two arrays (large)", test_lax_sort_two_arrays_larger()))

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
