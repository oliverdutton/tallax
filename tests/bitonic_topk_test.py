import functools
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from tallax._src.bitonic_topk import bitonic_topk, pallas_compatible_bitonic_topk, top1
from tallax._src.utils import is_cpu_platform
from tallax._src.test_utils import verify_topk_output


@pytest.mark.parametrize("shape", [(8, 128), (16, 256), (13, 167)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32])
def test_bitonic_topk_axis1(shape, dtype):
    """Test bitonic_topk for axis=1 (last axis)."""
    interpret = is_cpu_platform()
    key = jax.random.PRNGKey(0)

    if dtype == jnp.float32:
        arr = jax.random.normal(key, shape).astype(dtype)
    else:
        arr = jax.random.randint(key, shape, 0, 1000).astype(dtype)

    # Create indices array with column indices (for axis=1 operation)
    # For shape (8, 128), we want [[0, 1, 2, ..., 127], [0, 1, 2, ..., 127], ...]
    indices = jnp.broadcast_to(jnp.arange(shape[1])[None, :], shape).astype(jnp.int32)

    k = min(128, shape[1])  # NUM_LANES or dimension size, whichever is smaller
    # On CPU, call pallas_compatible_bitonic_topk directly (Pallas causes segfaults)
    # On TPU/GPU, use the full bitonic_topk with Pallas
    if interpret:
        result_values, result_indices = pallas_compatible_bitonic_topk([arr, indices], k=k, num_keys=1)
    else:
        result_values, result_indices = bitonic_topk([arr, indices], k=k, num_keys=1, descending=True, interpret=interpret)

    # Verify using test_utils (axis=1 is default)
    valid = verify_topk_output(arr, (result_values, result_indices), axis=1)
    assert valid.all(), f"Top-k validation failed for shape {shape}, dtype {dtype}"


@pytest.mark.parametrize("shape", [(8, 128), (16, 256), (128, 8), (256, 16)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32])
@pytest.mark.parametrize("axis", [0, 1])
def test_top1_pallas(shape, dtype, axis):
    """Test top1 wrapped in pallas kernel for both axes. Note: top1 requires dim0 to be a power of 2."""
    interpret = is_cpu_platform()
    key = jax.random.PRNGKey(1 + axis)  # Different seed per axis

    if dtype == jnp.float32:
        arr = jax.random.normal(key, shape).astype(dtype)
    else:
        arr = jax.random.randint(key, shape, 0, 1000).astype(dtype)

    # Create indices array
    # axis=0: row indices [[0,0,...], [1,1,...], ...]
    # axis=1: column indices [[0,1,2,...], [0,1,2,...], ...]
    if axis == 0:
        indices = jnp.broadcast_to(jnp.arange(shape[0])[:, None], shape).astype(jnp.int32)
        # top1 returns 1D output with shape (batch_size,) where batch_size = shape[1] for axis=0
        out_shape_1d = (shape[1],)
    else:  # axis == 1
        indices = jnp.broadcast_to(jnp.arange(shape[1])[None, :], shape).astype(jnp.int32)
        # top1 returns 1D output with shape (batch_size,) where batch_size = shape[0] for axis=1
        out_shape_1d = (shape[0],)

    def top1_kernel(values_ref, indices_ref, out_values_ref, out_indices_ref):
        """Top1 kernel."""
        result_values, result_indices = top1(
            [values_ref[...], indices_ref[...]],
            num_keys=1,
            axis=axis
        )
        # top1 now returns 1D outputs directly
        out_values_ref[...] = result_values
        out_indices_ref[...] = result_indices

    @functools.partial(jax.jit, static_argnames=("interpret",))
    def top1_pallas(values, indices, interpret=False):
        return pl.pallas_call(
            top1_kernel,
            out_shape=[
                jax.ShapeDtypeStruct(out_shape_1d, values.dtype),
                jax.ShapeDtypeStruct(out_shape_1d, jnp.int32),
            ],
            interpret=interpret
        )(values, indices)

    result_values, result_indices = top1_pallas(arr, indices, interpret=interpret)

    # Verify using axis parameter - verify_topk_output handles 1D outputs directly
    valid = verify_topk_output(arr, (result_values, result_indices), axis=axis)
    assert valid.all(), f"Top1 validation failed for shape {shape}, dtype {dtype}, axis={axis}"
