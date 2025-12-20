import pytest
import jax
import jax.numpy as jnp
from tallax import tax
from tallax._src.utils import is_cpu_platform
from tallax._src.test_utils import verify_topk_output


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
    recall = verify_topk_output(operand, outputs, axis=1, approximate=True)

    assert (recall > 0.95).all(), (
        f"approx_max_k validation failed for shape {shape}, dtype {dtype}, k={k}: "
        f"recall={recall.mean():.3f}"
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
    recall = verify_topk_output(operand, outputs, axis=1, approximate=True)

    assert (recall > 0.9).all(), (
        f"approx_max_k large vocab validation failed for dtype {dtype}, k={k}: "
        f"recall={recall.mean():.3f}"
    )
