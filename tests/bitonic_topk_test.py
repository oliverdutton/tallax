import functools
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from tallax.tax.bitonic_topk import bitonic_topk, top1
from tallax.utils import is_cpu_platform


@pytest.mark.parametrize("shape", [(8, 128), (16, 256)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32])
def test_bitonic_topk_axis1(shape, dtype):
    """Test bitonic_topk for axis=1 (last axis)."""
    interpret = is_cpu_platform()
    key = jax.random.PRNGKey(0)

    if dtype == jnp.float32:
        arr = jax.random.normal(key, shape).astype(dtype)
    else:
        arr = jax.random.randint(key, shape, 0, 1000).astype(dtype)

    k = 128  # NUM_LANES
    # bitonic_topk returns top-k in descending order along last axis
    result_values, result_indices = bitonic_topk(arr, k=k, num_keys=1, descending=True, interpret=interpret)

    # Expected: sort along axis 1 in descending order and take top k
    sorted_indices = jnp.argsort(arr, axis=1, descending=True)[:, :k]
    expected_values = jnp.take_along_axis(arr, sorted_indices, axis=1)

    if dtype == jnp.float32:
        np.testing.assert_allclose(result_values, expected_values, rtol=1e-5)
    else:
        np.testing.assert_array_equal(result_values, expected_values)


@pytest.mark.parametrize("shape", [(8, 128), (16, 256)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32])
def test_top1_axis0_pallas(shape, dtype):
    """Test top1 for axis=0 wrapped in pallas kernel."""
    interpret = is_cpu_platform()
    key = jax.random.PRNGKey(1)

    if dtype == jnp.float32:
        arr = jax.random.normal(key, shape).astype(dtype)
    else:
        arr = jax.random.randint(key, shape, 0, 1000).astype(dtype)

    # Create indices array
    indices = jnp.arange(shape[0] * shape[1]).reshape(shape).astype(jnp.int32)

    def top1_kernel(values_ref, indices_ref, out_values_ref, out_indices_ref):
        """Top1 kernel for axis=0."""
        result_values, result_indices = top1(
            [values_ref[...], indices_ref[...]],
            num_keys=1,
            axis=0
        )
        out_values_ref[...] = result_values
        out_indices_ref[...] = result_indices

    @functools.partial(jax.jit, static_argnames=("interpret",))
    def top1_pallas(values, indices, interpret=False):
        return pl.pallas_call(
            top1_kernel,
            out_shape=[
                jax.ShapeDtypeStruct((1, shape[1]), values.dtype),
                jax.ShapeDtypeStruct((1, shape[1]), jnp.int32),
            ],
            interpret=interpret
        )(values, indices)

    result_values, result_indices = top1_pallas(arr, indices, interpret=interpret)

    # Expected: argmax along axis 0
    expected_max_indices = jnp.argmax(arr, axis=0, keepdims=True)
    expected_values = jnp.take_along_axis(arr, expected_max_indices, axis=0)

    if dtype == jnp.float32:
        np.testing.assert_allclose(result_values, expected_values, rtol=1e-5)
    else:
        np.testing.assert_array_equal(result_values, expected_values)
