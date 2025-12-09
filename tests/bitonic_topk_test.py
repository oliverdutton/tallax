import functools
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from tallax.tax.bitonic_topk import bitonic_topk, pallas_compatible_bitonic_topk, top1
from tallax.utils import is_cpu_platform
from tallax.test_utils import verify_topk_output


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

    # Verify using test_utils
    valid = verify_topk_output(arr, (result_values, result_indices))
    assert valid.all(), f"Top-k validation failed for shape {shape}, dtype {dtype}"


@pytest.mark.parametrize("shape", [(8, 128), (16, 256), (128, 8), (256, 16)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32])
def test_top1_axis0_pallas(shape, dtype):
    """Test top1 for axis=0 wrapped in pallas kernel. Note: top1 requires dim0 to be a power of 2."""
    interpret = is_cpu_platform()
    key = jax.random.PRNGKey(1)

    if dtype == jnp.float32:
        arr = jax.random.normal(key, shape).astype(dtype)
    else:
        arr = jax.random.randint(key, shape, 0, 1000).astype(dtype)

    # Create indices array with row indices (for axis=0 operation)
    # For shape (8, 128), we want [[0, 0, ..., 0], [1, 1, ..., 1], ..., [7, 7, ..., 7]]
    indices = jnp.broadcast_to(jnp.arange(shape[0])[:, None], shape).astype(jnp.int32)

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

    # Transpose to match verify_topk_output expectations (batch x k)
    # arr is (dim0, dim1), we want to get top1 along dim0 for each column
    # So we transpose and verify each column independently
    arr_T = arr.T  # (dim1, dim0)
    result_values_T = result_values.T  # (dim1, 1)
    result_indices_T = result_indices.T  # (dim1, 1)

    # verify_topk_output expects (values, indices) where both are (batch, k)
    valid = verify_topk_output(arr_T, (result_values_T, result_indices_T))
    assert valid.all(), f"Top1 axis=0 validation failed for shape {shape}, dtype {dtype}"
