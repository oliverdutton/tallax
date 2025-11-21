import jax
import jax.numpy as jnp
import pytest
import numpy as np
from tallax import tumpy

def test_sort_correctness():
    key = jax.random.PRNGKey(0)
    shape = (2, 16)
    a = jax.random.randint(key, shape, 0, 100)

    expected = jnp.sort(a, axis=-1)
    result = tumpy.sort(a, axis=-1)

    np.testing.assert_array_equal(result, expected)

def test_argsort_correctness():
    key = jax.random.PRNGKey(1)
    shape = (2, 16)
    a = jax.random.randint(key, shape, 0, 100)

    # Force unique elements to ensure argsort is deterministic/checkable easily
    a = jax.random.permutation(key, jnp.arange(32)).reshape(shape)

    expected = jnp.argsort(a, axis=-1)
    result = tumpy.argsort(a, axis=-1)

    np.testing.assert_array_equal(result, expected)

def test_multidimensional_reshape():
    key = jax.random.PRNGKey(2)
    shape = (2, 2, 2, 16)

    a = jax.random.randint(key, shape, 0, 100)

    expected = jnp.sort(a, axis=-1)
    result = tumpy.sort(a, axis=-1)

    assert result.shape == shape
    np.testing.assert_array_equal(result, expected)

def test_1d_array():
    key = jax.random.PRNGKey(3)
    a = jax.random.randint(key, (16,), 0, 100)

    expected = jnp.sort(a, axis=-1)
    result = tumpy.sort(a, axis=-1)

    assert result.shape == (16,)
    np.testing.assert_array_equal(result, expected)

def test_descending():
    key = jax.random.PRNGKey(4)
    a = jax.random.randint(key, (2, 16), 0, 100)

    expected = jnp.sort(a, axis=-1, descending=True)
    result = tumpy.sort(a, axis=-1, descending=True)

    np.testing.assert_array_equal(result, expected)

    expected_args = jnp.argsort(a, axis=-1, descending=True)
    result_args = tumpy.argsort(a, axis=-1, descending=True)

    result_vals = jnp.take_along_axis(a, result_args, axis=-1)
    expected_vals = jnp.take_along_axis(a, expected_args, axis=-1)

    np.testing.assert_array_equal(result_vals, expected_vals)

def test_axis_validation():
    a = jnp.zeros((4, 4))

    # Valid axes
    tumpy.sort(a, axis=-1)
    tumpy.sort(a, axis=1)

    # Invalid axes
    with pytest.raises(ValueError, match="only supports sorting along the last axis"):
        tumpy.sort(a, axis=0)

    with pytest.raises(ValueError):
        tumpy.argsort(a, axis=0)

def test_kind_ignored():
    a = jnp.array([3, 1, 2])
    # Should run without error
    tumpy.sort(a, kind='mergesort')
    tumpy.argsort(a, kind='heapsort')

def test_order_parameter():
    key = jax.random.PRNGKey(5)
    a = jax.random.randint(key, (2, 16), 0, 100)

    # Test order='descending' implies descending=True
    expected = jnp.sort(a, axis=-1, descending=True)
    result = tumpy.sort(a, axis=-1, order='descending')

    np.testing.assert_array_equal(result, expected)

    expected_args = jnp.argsort(a, axis=-1, descending=True)
    result_args = tumpy.argsort(a, axis=-1, order='descending')

    result_vals = jnp.take_along_axis(a, result_args, axis=-1)
    expected_vals = jnp.take_along_axis(a, expected_args, axis=-1)
    np.testing.assert_array_equal(result_vals, expected_vals)

def test_axis_none_flatten():
    key = jax.random.PRNGKey(6)
    shape = (2, 16)
    a = jax.random.randint(key, shape, 0, 100)

    expected = jnp.sort(a, axis=None)
    result = tumpy.sort(a, axis=None)

    assert result.ndim == 1
    assert result.shape == (32,)
    np.testing.assert_array_equal(result, expected)

    expected_args = jnp.argsort(a, axis=None)
    result_args = tumpy.argsort(a, axis=None)

    # Argsort on flattened array returns indices into flattened array
    assert result_args.ndim == 1
    assert result_args.shape == (32,)

    # Verify values
    a_flat = a.ravel()
    result_vals = a_flat[result_args]
    expected_vals = a_flat[expected_args]
    np.testing.assert_array_equal(result_vals, expected_vals)
