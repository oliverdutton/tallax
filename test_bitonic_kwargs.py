"""Test script for bitonic_sort with new kwargs."""
import jax
import jax.numpy as jnp
from tallax._src.bitonic_topk import bitonic_sort, bitonic_sort_arrays


def test_bitonic_sort_16_1024():
    """Test bitonic_sort with shape (16, 1024) - first useful shape."""
    print("Testing shape (16, 1024)...")
    shape = (16, 1024)
    key = jax.random.PRNGKey(42)

    # Generate test data
    arr = jax.random.randint(key, shape, 0, 1000, dtype=jnp.int32)

    # Test basic sort
    print("  Testing basic sort...")
    result = bitonic_sort_arrays([arr], num_keys=1, descending=False)
    sorted_arr = result[0]

    # Verify shape
    assert sorted_arr.shape == shape, f"Shape mismatch: {sorted_arr.shape} != {shape}"

    # Verify each row is sorted
    for i in range(shape[0]):
        row = sorted_arr[i]
        is_sorted = jnp.all(row[:-1] <= row[1:])
        assert is_sorted, f"Row {i} is not sorted correctly"

    # Verify against reference
    expected = jnp.sort(arr, axis=1)
    assert jnp.allclose(sorted_arr, expected), "Output doesn't match reference"
    print("  ✓ Basic sort passed")

    # Test with tile_unroll=8
    print("  Testing with tile_unroll=8...")
    result = bitonic_sort_arrays(
        [arr], num_keys=1, descending=False, tile_unroll=8
    )
    sorted_arr = result[0]
    assert jnp.allclose(sorted_arr, expected), "tile_unroll=8 failed"
    print("  ✓ tile_unroll=8 passed")

    # Test with max_num_fused_stages
    print("  Testing with max_num_fused_stages=5...")
    result = bitonic_sort_arrays(
        [arr], num_keys=1, descending=False, max_num_fused_stages=5
    )
    sorted_arr = result[0]
    assert jnp.allclose(sorted_arr, expected), "max_num_fused_stages failed"
    print("  ✓ max_num_fused_stages=5 passed")

    # Test with unroll_stages
    print("  Testing with unroll_stages=True...")
    result = bitonic_sort_arrays(
        [arr], num_keys=1, descending=False, unroll_stages=True
    )
    sorted_arr = result[0]
    assert jnp.allclose(sorted_arr, expected), "unroll_stages failed"
    print("  ✓ unroll_stages=True passed")

    # Test with transpose_scratch_refs
    print("  Testing with transpose_scratch_refs=True...")
    result = bitonic_sort_arrays(
        [arr], num_keys=1, descending=False, transpose_scratch_refs=True
    )
    sorted_arr = result[0]
    assert jnp.allclose(sorted_arr, expected), "transpose_scratch_refs failed"
    print("  ✓ transpose_scratch_refs=True passed")

    # Test all kwargs together
    print("  Testing with all kwargs...")
    result = bitonic_sort_arrays(
        [arr],
        num_keys=1,
        descending=False,
        max_num_fused_stages=5,
        tile_unroll=8,
        unroll_stages=True,
        transpose_scratch_refs=True,
    )
    sorted_arr = result[0]
    assert jnp.allclose(sorted_arr, expected), "All kwargs together failed"
    print("  ✓ All kwargs together passed")

    print("✓ All tests for (16, 1024) passed!\n")


def test_descending():
    """Test descending sort."""
    print("Testing descending sort with (16, 1024)...")
    shape = (16, 1024)
    key = jax.random.PRNGKey(123)
    arr = jax.random.randint(key, shape, 0, 1000, dtype=jnp.int32)

    result = bitonic_sort_arrays(
        [arr], num_keys=1, descending=True, tile_unroll=8
    )
    sorted_arr = result[0]

    # Verify descending order
    for i in range(shape[0]):
        row = sorted_arr[i]
        is_sorted = jnp.all(row[:-1] >= row[1:])
        assert is_sorted, f"Row {i} is not sorted in descending order"

    expected = jnp.sort(arr, axis=1)[:, ::-1]
    assert jnp.allclose(sorted_arr, expected), "Descending sort failed"
    print("✓ Descending sort passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing bitonic_sort with new kwargs")
    print("=" * 60)
    print()

    test_bitonic_sort_16_1024()
    test_descending()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
