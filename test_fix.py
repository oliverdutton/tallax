"""Test the fix for the sort segfault bug."""
import jax
import jax.numpy as jnp
from tallax import tax

def test_float32_return_argsort_interpret():
    """Test float32 with return_argsort in interpret mode (previously segfaulted)."""
    print("Test 1: float32 + return_argsort + interpret=True")
    print("        (This previously caused segfault)")

    key = jax.random.key(0)
    x = jax.random.normal(key, (16, 128), dtype=jnp.float32)

    try:
        result = tax.sort([x], num_keys=1, return_argsort=True, interpret=True)
        print(f"✓ SUCCESS: Shapes = {[r.shape for r in result]}")
        print(f"  Result dtypes: {[r.dtype for r in result]}")
        # Verify results are correct
        assert result[0].shape == (16, 128), "Values shape mismatch"
        assert result[1].shape == (16, 128), "Indices shape mismatch"
        assert result[0].dtype == jnp.float32, "Values dtype should be float32"
        assert result[1].dtype == jnp.int32, "Indices dtype should be int32"
        print("  ✓ All assertions passed\n")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_float32_no_argsort_interpret():
    """Test float32 without return_argsort in interpret mode (baseline)."""
    print("Test 2: float32 + no return_argsort + interpret=True")
    print("        (This always worked)")

    key = jax.random.key(0)
    x = jax.random.normal(key, (16, 128), dtype=jnp.float32)

    try:
        result = tax.sort([x], num_keys=1, return_argsort=False, interpret=True)
        print(f"✓ SUCCESS: Shape = {result[0].shape}")
        assert result[0].shape == (16, 128)
        print("  ✓ Assertion passed\n")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        return False

def test_bfloat16_return_argsort_interpret():
    """Test bfloat16 with return_argsort in interpret mode (baseline)."""
    print("Test 3: bfloat16 + return_argsort + interpret=True")
    print("        (This always worked due to packing)")

    key = jax.random.key(0)
    x = jax.random.normal(key, (16, 128), dtype=jnp.float32).astype(jnp.bfloat16)

    try:
        result = tax.sort([x], num_keys=1, return_argsort=True, interpret=True)
        print(f"✓ SUCCESS: Shapes = {[r.shape for r in result]}")
        print(f"  Result dtypes: {[r.dtype for r in result]}")
        assert result[0].dtype == jnp.bfloat16
        assert result[1].dtype == jnp.int32
        print("  ✓ Assertions passed\n")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        return False

def test_multiple_arrays_interpret():
    """Test sorting multiple arrays (float32 + float32) with indices."""
    print("Test 4: Two float32 arrays + return_argsort + interpret=True")
    print("        (Edge case)")

    key = jax.random.key(0)
    keys = jax.random.split(key, 2)
    x1 = jax.random.normal(keys[0], (16, 128), dtype=jnp.float32)
    x2 = jax.random.normal(keys[1], (16, 128), dtype=jnp.float32)

    try:
        result = tax.sort([x1, x2], num_keys=2, return_argsort=True, interpret=True)
        print(f"✓ SUCCESS: Shapes = {[r.shape for r in result]}")
        print(f"  Result dtypes: {[r.dtype for r in result]}")
        assert len(result) == 3  # x1, x2, indices
        print("  ✓ Assertions passed\n")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Sort Segfault Fix")
    print("=" * 70)
    print()

    results = []

    # The critical test - this should now work!
    results.append(("float32 + return_argsort + interpret", test_float32_return_argsort_interpret()))

    # Baseline tests that should continue to work
    results.append(("float32 + no argsort + interpret", test_float32_no_argsort_interpret()))
    results.append(("bfloat16 + return_argsort + interpret", test_bfloat16_return_argsort_interpret()))
    results.append(("two arrays + return_argsort + interpret", test_multiple_arrays_interpret()))

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")

    if all(passed for _, passed in results):
        print("\n🎉 All tests passed! The fix works!")
        exit(0)
    else:
        print("\n❌ Some tests failed")
        exit(1)
