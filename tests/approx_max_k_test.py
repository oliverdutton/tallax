import functools
import pytest
import jax
import jax.numpy as jnp
from tallax import tax
from tallax._src.utils import is_cpu_platform


def verify_approx_topk_output(x, outs, k, axis=-1, recall_target=0.95):
    """Validate approximate top-k outputs.

    Args:
        x: Input array (must be 2D)
        outs: Tuple of (values, indices) from approx_max_k (both must be 2D)
        k: Number of top elements requested
        axis: Axis along which top-k was computed (default -1)
        recall_target: Minimum fraction of values >= threshold (default 0.95)

    Returns:
        Boolean array indicating validity for each batch element
    """
    if x.ndim != 2:
        raise ValueError(f"verify_approx_topk_output only supports 2D inputs, got {x.ndim}D")

    out_vals, out_indexs = outs

    if out_vals.ndim != 2 or out_indexs.ndim != 2:
        raise ValueError(f"verify_approx_topk_output requires 2D outputs, got values.ndim={out_vals.ndim}, indices.ndim={out_indexs.ndim}")

    axis = axis % 2
    batch_axis = 1 - axis

    @functools.partial(jax.vmap, in_axes=batch_axis)
    def verify_slice(x_slice, vals_slice, idxs_slice):
        threshold = jax.lax.top_k(x_slice, k)[0][-1]
        n = len(x_slice)
        valid = True

        valid &= jnp.mean(vals_slice >= threshold) > recall_target
        valid &= (x_slice[idxs_slice] == vals_slice).all()

        i = jnp.unique(idxs_slice, size=k, fill_value=-1)
        valid &= ((i >= 0) & (i < n)).all()
        return valid

    return verify_slice(x, out_vals, out_indexs)


@pytest.mark.parametrize("shape", [(8, 128), (16, 256), (32, 512), (64, 1024)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32])
@pytest.mark.parametrize("k", [10, 32, 64])
@pytest.mark.skipif(is_cpu_platform(), reason="approx_max_k tests require TPU/GPU")
def test_approx_max_k_2d(shape, dtype, k):
    """Test approx_max_k with 2D inputs and axis=-1."""
    if k >= shape[1]:
        pytest.skip(f"k={k} >= vocab_size={shape[1]}")

    key = jax.random.key(42)
    operand = jax.random.normal(key, shape, dtype=dtype) if dtype == jnp.float32 else jax.random.randint(key, shape, 0, 1000, dtype=dtype)

    outputs = tax.approx_max_k(operand, k=k, reduction_dimension=-1)
    validation = verify_approx_topk_output(operand, outputs, k=k, axis=-1)

    assert validation.all(), (
        f"approx_max_k validation failed for shape {shape}, dtype {dtype}, k={k}: "
        f"{int(validation.sum())}/{shape[0]} rows passed"
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32])
@pytest.mark.parametrize("k", [32, 64, 128])
@pytest.mark.skipif(is_cpu_platform(), reason="approx_max_k tests require TPU/GPU")
def test_approx_max_k_large_vocab(dtype, k):
    """Test approx_max_k with (128, 8192) iota reshaped."""
    shape = (128, 8192)

    operand = jnp.arange(shape[0] * shape[1], dtype=dtype).reshape(shape)
    if dtype == jnp.float32:
        operand = operand + jax.random.normal(jax.random.key(123), shape, dtype=dtype) * 0.1

    outputs = tax.approx_max_k(operand, k=k, reduction_dimension=-1)
    validation = verify_approx_topk_output(operand, outputs, k=k, axis=-1, recall_target=0.9)

    assert validation.all(), (
        f"approx_max_k large vocab validation failed for dtype {dtype}, k={k}: "
        f"{int(validation.sum())}/{shape[0]} rows passed"
    )
