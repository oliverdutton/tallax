import functools
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from tallax._src.bitonic_topk import bitonic_topk, bitonic_topk_arrays, max_arrays
from tallax._src.utils import is_cpu_platform
from tallax._src.test_utils import verify_topk_output


@pytest.mark.parametrize("shape", [(8, 128), (16, 256), (13, 167), (256, 256), (173, 195)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32])
def test_bitonic_topk_axis1(shape, dtype):
    """Test bitonic_topk for axis=1 (last axis)."""
    interpret = is_cpu_platform()
    key = jax.random.PRNGKey(0)

    if dtype == jnp.float32:
        arr = jax.random.normal(key, shape).astype(dtype)
    else:
        arr = jax.random.randint(key, shape, 0, 1000).astype(dtype)

    indices = jax.lax.broadcasted_iota(jnp.int32, shape, 1)

    k = min(128, shape[1])  # NUM_LANES or dimension size, whichever is smaller
    # On CPU, call bitonic_topk_arrays directly (Pallas causes segfaults)
    # On TPU/GPU, use the full bitonic_topk with Pallas
    if interpret:
        result_values, result_indices = bitonic_topk_arrays([arr, indices], k=k, num_keys=1)
    else:
        result_values, result_indices = bitonic_topk([arr, indices], k=k, num_keys=1, descending=True, interpret=interpret)

    # Verify using test_utils (axis=1 is default)
    valid = verify_topk_output(arr, (result_values, result_indices), axis=1)
    assert valid.all(), f"Top-k validation failed for shape {shape}, dtype {dtype}"


@pytest.mark.parametrize("shape", [(8, 128), (16, 256), (128, 8), (256, 16), (256, 256), (173, 195)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32])
@pytest.mark.parametrize("axis", [0, 1])
def test_top1_pallas(shape, dtype, axis):
    """Test top1 wrapped in pallas kernel for both axes."""
    interpret = is_cpu_platform()
    key = jax.random.PRNGKey(1 + axis)  # Different seed per axis

    if dtype == jnp.float32:
        arr = jax.random.normal(key, shape).astype(dtype)
    else:
        arr = jax.random.randint(key, shape, 0, 1000).astype(dtype)

    if axis == 0:
        indices = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
        # top1 returns 1D output with shape (batch_size,) where batch_size = shape[1] for axis=0
        out_shape_1d = (shape[1],)
    else:  # axis == 1
        indices = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
        # top1 returns 1D output with shape (batch_size,) where batch_size = shape[0] for axis=1
        out_shape_1d = (shape[0],)

    def top1_refs(values_ref, indices_ref, out_values_ref, out_indices_ref):
        """Top1 refs kernel."""
        result_values, result_indices = max_arrays(
            [values_ref[...], indices_ref[...]],
            num_keys=1,
            axis=axis
        )
        # max_arrays now returns 1D outputs directly
        out_values_ref[...] = result_values
        out_indices_ref[...] = result_indices

    @functools.partial(jax.jit, static_argnames=("interpret",))
    def top1_pallas(values, indices, interpret=False):
        return pl.pallas_call(
            top1_refs,
            out_shape=[
                jax.ShapeDtypeStruct(out_shape_1d, values.dtype),
                jax.ShapeDtypeStruct(out_shape_1d, jnp.int32),
            ],
            interpret=interpret
        )(values, indices)

    result_values, result_indices = top1_pallas(arr, indices, interpret=interpret)

    # Reshape 1D outputs to 2D for verify_topk_output
    if axis == 0:
        # axis=0: result is (shape[1],) -> reshape to (1, shape[1])
        result_values = jnp.expand_dims(result_values, axis=0)
        result_indices = jnp.expand_dims(result_indices, axis=0)
    else:  # axis == 1
        # axis=1: result is (shape[0],) -> reshape to (shape[0], 1)
        result_values = jnp.expand_dims(result_values, axis=1)
        result_indices = jnp.expand_dims(result_indices, axis=1)

    # Verify using axis parameter with 2D outputs
    valid = verify_topk_output(arr, (result_values, result_indices), axis=axis)
    assert valid.all(), f"Top1 validation failed for shape {shape}, dtype {dtype}, axis={axis}"
