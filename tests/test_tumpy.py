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
