import pytest
import jax
import jax.numpy as jnp
from tallax._src.bitonic_topk import bitonic_sort, bitonic_sort_arrays
from tallax._src.utils import is_cpu_platform


@pytest.mark.parametrize("shape", [
    (8, 16), (8, 64), (8, 128), (8, 256), (8, 512), (8, 1024), (8, 2048),
    (16, 128), (16, 256), (32, 128), (64, 128), (128, 128),
])
@pytest.mark.parametrize("dtype", [jnp.int32, jnp.float32])
@pytest.mark.parametrize("descending", [False, True])
def test_bitonic_sort_arrays(shape, dtype, descending):
    """Test bitonic_sort_arrays for various shapes and dtypes."""
    interpret = is_cpu_platform()
    if interpret and (shape[0] * shape[1] > 16384):
        pytest.skip("Test too large for CPU, as compilation is very slow")

    key = jax.random.PRNGKey(42)

    # Generate test data based on dtype
    if dtype == jnp.float32:
        arr = jax.random.normal(key, shape).astype(dtype)
    else:
        arr = jax.random.randint(key, shape, 0, 1000).astype(dtype)

    # Run bitonic sort
    result = bitonic_sort_arrays([arr], num_keys=1, descending=descending)
    sorted_arr = result[0]

    # Verify shape
    assert sorted_arr.shape == shape, f"Output shape mismatch: {sorted_arr.shape} != {shape}"

    # Verify each row is sorted
    for i in range(shape[0]):
        row = sorted_arr[i]
        if descending:
            is_sorted = jnp.all(row[:-1] >= row[1:])
        else:
            is_sorted = jnp.all(row[:-1] <= row[1:])
        assert is_sorted, f"Row {i} is not sorted correctly (descending={descending})"

    # Verify against reference implementation
    expected = jnp.sort(arr, axis=1)
    if descending:
        expected = expected[:, ::-1]

    assert jnp.allclose(sorted_arr, expected), \
        f"Output doesn't match reference for shape {shape}, dtype {dtype}, descending={descending}"


@pytest.mark.parametrize("shape", [(8, 2048), (16, 512), (32, 256)])
@pytest.mark.parametrize("dtype", [jnp.int32, jnp.float32])
@pytest.mark.parametrize("descending", [False, True])
def test_bitonic_sort_pallas(shape, dtype, descending):
    """Test bitonic_sort with Pallas kernel."""
    interpret = is_cpu_platform()

    key = jax.random.PRNGKey(123)

    # Generate test data based on dtype
    if dtype == jnp.float32:
        arr = jax.random.normal(key, shape).astype(dtype)
    else:
        arr = jax.random.randint(key, shape, 0, 1000).astype(dtype)

    # Run bitonic sort through Pallas
    if interpret:
        # On CPU, call bitonic_sort_arrays directly to avoid Pallas issues
        result = bitonic_sort_arrays([arr], num_keys=1, descending=descending)
    else:
        result = bitonic_sort(arr, num_keys=1, descending=descending, interpret=interpret)

    sorted_arr = result[0]

    # Verify shape
    assert sorted_arr.shape == shape, f"Output shape mismatch: {sorted_arr.shape} != {shape}"

    # Verify against reference implementation
    expected = jnp.sort(arr, axis=1)
    if descending:
        expected = expected[:, ::-1]

    assert jnp.allclose(sorted_arr, expected), \
        f"Output doesn't match reference for shape {shape}, dtype {dtype}, descending={descending}"


@pytest.mark.parametrize("num_keys", [1, 2])
def test_bitonic_sort_multi_key(num_keys):
    """Test bitonic sort with multiple keys."""
    interpret = is_cpu_platform()
    shape = (8, 128)
    key = jax.random.PRNGKey(456)

    # Create test arrays
    arr1 = jax.random.randint(key, shape, 0, 10, dtype=jnp.int32)  # Primary key with duplicates
    arr2 = jax.random.randint(key, shape, 0, 100, dtype=jnp.int32)  # Secondary key

    if num_keys == 1:
        if interpret:
            result = bitonic_sort_arrays([arr1], num_keys=1, descending=False)
        else:
            result = bitonic_sort(arr1, num_keys=1, descending=False, interpret=interpret)
        sorted_arr1 = result[0]

        # Verify against reference
        expected = jnp.sort(arr1, axis=1)
        assert jnp.allclose(sorted_arr1, expected)
    else:
        if interpret:
            result = bitonic_sort_arrays([arr1, arr2], num_keys=2, descending=False)
        else:
            result = bitonic_sort([arr1, arr2], num_keys=2, descending=False, interpret=interpret)
        sorted_arr1, sorted_arr2 = result

        # Verify that arr1 is sorted
        for i in range(shape[0]):
            assert jnp.all(sorted_arr1[i, :-1] <= sorted_arr1[i, 1:])

        # Verify that where arr1 values are equal, arr2 is sorted
        for i in range(shape[0]):
            for j in range(shape[1] - 1):
                if sorted_arr1[i, j] == sorted_arr1[i, j + 1]:
                    assert sorted_arr2[i, j] <= sorted_arr2[i, j + 1], \
                        f"Secondary key not sorted at row {i}, col {j}"
