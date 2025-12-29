import pytest
import jax
import jax.numpy as jnp
from tallax.divide_and_filter_topk.topk import topk
from tallax.tax.utils import is_cpu_platform
from tallax.tax.test_utils import verify_topk_output


@pytest.mark.parametrize("shape", [(8, 128), (16, 256), (13, 167), (256, 256), (173, 195), (16, 16384), (13, 11571)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32])
@pytest.mark.skipif(
    is_cpu_platform(),
    reason="Divide and filter top-k tests require TPU/GPU - CPU uses interpret mode which is slow"
)
def test_divide_and_filter_topk(shape, dtype):
    """Test divide and filter top-k implementation with exact match validation."""
    k = min(137, shape[1] // 2)

    # Generate test data
    key = jax.random.key(0)
    if dtype == jnp.float32:
        logits = jax.random.normal(key, shape, dtype=dtype)
    else:
        logits = jax.random.randint(key, shape, 0, 1000, dtype=dtype)

    # Run divide and filter top-k implementation
    outputs = topk(
        logits,
        k=k,
        interpret=is_cpu_platform(),
        num_bins=128 if shape[1] <= 128 else 256)

    # Validate results using verify_topk_output (axis=1 is default)
    validation = verify_topk_output(logits, outputs, axis=1)

    assert validation.all(), (
        f"Divide and filter top-k validation failed for shape {shape}, dtype {dtype}: "
        f"{int(validation.sum())}/{shape[0]} rows passed"
    )


# tests the merging unconverged bins logic
@pytest.mark.parametrize("shape", [(16, 16384), (13, 11571)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32])
@pytest.mark.parametrize("k", [17, 128, 157])
@pytest.mark.skipif(
    is_cpu_platform(),
    reason="Divide and filter top-k tests require TPU/GPU - CPU uses interpret mode which is slow"
)
def test_divide_and_filter_topk_worst_case_values(shape, dtype, k):
    """Test divide and filter top-k implementation with exact match validation."""
    # Generate test data
    key = jax.random.key(0)
    if dtype == jnp.float32:
        logits = jax.random.normal(key, shape, dtype=dtype)
    else:
        logits = jax.random.randint(key, shape, 0, 1000, dtype=dtype)
    
    # organize that topk is all in bin 19
    logits = logits.at[:,19::256].add(1000)

    # Run divide and filter top-k implementation
    outputs = topk(
        logits,
        k=k,
        interpret=is_cpu_platform(),
        num_bins=256)

    # Validate results using verify_topk_output (axis=1 is default)
    validation = verify_topk_output(logits, outputs, axis=1)

    assert validation.all(), (
        f"Divide and filter top-k validation failed for shape {shape}, dtype {dtype}: "
        f"{int(validation.sum())}/{shape[0]} rows passed"
    )
