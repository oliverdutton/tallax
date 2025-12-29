import pytest
import jax
import jax.numpy as jnp
from tallax import tax
from tallax.tax.utils import is_cpu_platform
from tallax.tax.test_utils import verify_topk_output


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32])
@pytest.mark.parametrize("k", [1, 2, 3, 17, 32, 64, 128])
@pytest.mark.parametrize("recall_target", [0.2, 0.5, 0.8, 0.95, 0.99])
@pytest.mark.parametrize("shape", [(128, 8192)])
@pytest.mark.parametrize("seed", [0, 42, 467])
@pytest.mark.skipif(is_cpu_platform(), reason="approx_max_k tests require TPU/GPU")
def test_approx_max_k(dtype, k, recall_target, shape, use_lax_approx_max_k_algorithm, seed):
    """Test approx_max_k."""
    key = jax.random.key(seed)
    if dtype == jnp.float32:
        operand = jax.random.normal(key, shape, dtype=dtype)
    else:
        operand = jax.random.randint(key, shape, 0, 2**24, dtype=dtype)

    outputs = tax.approx_max_k(operand, k=k, recall_target=recall_target)
    recall = verify_topk_output(operand, outputs, axis=1, approximate=True).mean()
    # you'd expect checking against recall target to fail 50% of the time, but in practice the approximation bounds are quite strong so this passes 100% of current tests happily
    test_recall_threshold = recall_target
    print(f'recall of {recall:.3f} for target {recall_target:.3f}')
    if k==1:
        assert recall==1., f"For k=1 recall must be perfect, but found {recall:.3f}"
    assert (recall > test_recall_threshold), (
        f"approx_max_k validation failed for dtype {dtype}, k={k}: "
        f"recall={recall:.3f}"
    )
