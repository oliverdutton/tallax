import jax
import jax.numpy as jnp
import pytest
import numpy as np
from tallax import tumpy as tnp
from tallax.utils import is_cpu_platform

def test_sort():
    interpret = is_cpu_platform()
    key = jax.random.PRNGKey(0)
    shape = (2, 16)
    a = jax.random.randint(key, shape, 0, 100)

    expected = jnp.sort(a, axis=-1)
    result = tnp.sort(a, axis=-1, interpret=interpret)

    np.testing.assert_array_equal(result, expected)

def test_argsort():
    interpret = is_cpu_platform()
    key = jax.random.PRNGKey(1)
    shape = (2, 16)
    a = jax.random.randint(key, shape, 0, 100)

    # Force unique elements to ensure argsort is deterministic/checkable easily
    a = jax.random.permutation(key, jnp.arange(32)).reshape(shape)

    expected = jnp.argsort(a, axis=-1)
    result = tnp.argsort(a, axis=-1, interpret=interpret)

    np.testing.assert_array_equal(result, expected)

def test_multidimensional_reshape():
    interpret = is_cpu_platform()
    key = jax.random.PRNGKey(2)
    shape = (2, 2, 2, 16)

    a = jax.random.randint(key, shape, 0, 100)

    expected = jnp.sort(a, axis=-1)
    result = tnp.sort(a, axis=-1, interpret=interpret)

    assert result.shape == shape
    np.testing.assert_array_equal(result, expected)

def test_1d_array():
    interpret = is_cpu_platform()
    key = jax.random.PRNGKey(3)
    a = jax.random.randint(key, (16,), 0, 100)

    expected = jnp.sort(a, axis=-1)
    result = tnp.sort(a, axis=-1, interpret=interpret)

    assert result.shape == (16,)
    np.testing.assert_array_equal(result, expected)

def test_descending():
    interpret = is_cpu_platform()
    key = jax.random.PRNGKey(4)
    a = jax.random.randint(key, (2, 16), 0, 100)

    expected = jnp.sort(a, axis=-1, descending=True)
    result = tnp.sort(a, axis=-1, descending=True, interpret=interpret)

    np.testing.assert_array_equal(result, expected)

    expected_args = jnp.argsort(a, axis=-1, descending=True)
    result_args = tnp.argsort(a, axis=-1, descending=True, interpret=interpret)

    result_vals = jnp.take_along_axis(a, result_args, axis=-1)
    expected_vals = jnp.take_along_axis(a, expected_args, axis=-1)

    np.testing.assert_array_equal(result_vals, expected_vals)

def test_axis_validation():
    interpret = is_cpu_platform()
    a = jnp.zeros((4, 4))

    # Valid axes
    tnp.sort(a, axis=-1, interpret=interpret)
    tnp.sort(a, axis=1, interpret=interpret)

    # Invalid axes
    with pytest.raises(ValueError, match="only supports sorting along the last axis"):
        tnp.sort(a, axis=0, interpret=interpret)

    with pytest.raises(ValueError):
        tnp.argsort(a, axis=0, interpret=interpret)

def test_kind_and_order_ignored():
    interpret = is_cpu_platform()
    a = jnp.array([3, 1, 2])
    # Should run without error
    tnp.sort(a, kind='mergesort', interpret=interpret)
    tnp.argsort(a, kind='heapsort', interpret=interpret)

    # Order should also be ignored (not raise error)
    tnp.sort(a, order='descending', interpret=interpret)
    tnp.argsort(a, order='something', interpret=interpret)

def test_axis_none_flatten():
    interpret = is_cpu_platform()
    key = jax.random.PRNGKey(6)
    shape = (2, 16)
    a = jax.random.randint(key, shape, 0, 100)

    expected = jnp.sort(a, axis=None)
    result = tnp.sort(a, axis=None, interpret=interpret)

    assert result.ndim == 1
    assert result.shape == (32,)
    np.testing.assert_array_equal(result, expected)

    expected_args = jnp.argsort(a, axis=None)
    result_args = tnp.argsort(a, axis=None, interpret=interpret)

    # Argsort on flattened array returns indices into flattened array
    assert result_args.ndim == 1
    assert result_args.shape == (32,)

    # Verify values
    a_flat = a.ravel()
    result_vals = a_flat[result_args]
    expected_vals = a_flat[expected_args]
    np.testing.assert_array_equal(result_vals, expected_vals)

@pytest.mark.parametrize("shape", [(8, 128), (16, 256)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32])
@pytest.mark.parametrize("axis", [0, 1])
def test_take_along_axis(shape, dtype, axis):
    interpret = is_cpu_platform()
    key = jax.random.PRNGKey(7)
    key_vals, key_idxs = jax.random.split(key)

    if dtype == jnp.float32:
        values = jax.random.normal(key_vals, shape).astype(dtype)
    else:
        values = jax.random.randint(key_vals, shape, 0, 100).astype(dtype)

    # Create indices for the specified axis
    indices_shape = list(shape)
    indices_shape[axis] = min(shape[axis], 64)  # Take fewer elements along axis
    indices = jax.random.randint(key_idxs, indices_shape, 0, shape[axis])

    # Expected result using jax.numpy
    expected = jnp.take_along_axis(values, indices, axis=axis)

    # Run Pallas take_along_axis
    result = tnp.take_along_axis(values, indices, axis=axis, interpret=interpret)

    if dtype == jnp.float32:
        np.testing.assert_allclose(result, expected)
    else:
        np.testing.assert_array_equal(result, expected)
