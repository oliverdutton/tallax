import pytest
import jax
import jax.numpy as jnp
from tallax import tax
from tallax._src.utils import is_cpu_platform
from tallax._src.test_utils import verify_topk_output


@pytest.mark.parametrize("shape", [(8, 128), (16, 256), (13, 167), (256, 256), (173, 195)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32])
@pytest.mark.skipif(
    is_cpu_platform(),
    reason="Divide and filter top-k tests require TPU/GPU - CPU uses interpret mode which is slow"
)
def test_divide_and_filter_topk(shape, dtype):
    """Test divide and filter top-k implementation with exact match validation."""
    k = min(128, shape[1] // 2)

    # Generate test data
    key = jax.random.key(0)
    if dtype == jnp.float32:
        logits = jax.random.normal(key, shape, dtype=dtype)
    else:
        logits = jax.random.randint(key, shape, 0, 1000, dtype=dtype)

    # Run divide and filter top-k implementation
    result = tax.top_k(logits, k=k, block_size=8, interpret=False)

    # Validate results using verify_topk_output (axis=1 is default)
    validation = verify_topk_output(logits, result, axis=1)

    assert validation.all(), (
        f"Divide and filter top-k validation failed for shape {shape}, dtype {dtype}: "
        f"{int(validation.sum())}/{shape[0]} rows passed"
    )
