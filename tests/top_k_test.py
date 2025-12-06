import jax
import jax.numpy as jnp
import pytest

from tallax import tax
from tallax.utils import is_cpu_platform
from tallax.test_utils import verify_topk_output


@pytest.mark.skipif(
    is_cpu_platform(),
    reason="Top-k tests require TPU/GPU - CPU uses interpret mode which is slow"
)
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_top_k(dtype):
    """Test top_k Pallas implementation with bfloat16 and float32."""
    num_queries = 16
    vocab_size = 201088
    k = 64

    # Generate test data
    key = jax.random.key(0)
    if dtype == jnp.bfloat16:
        logits = jax.random.normal(
            key, (num_queries, vocab_size), dtype=jnp.float32
        ).astype(jnp.bfloat16)
    else:
        logits = jax.random.normal(
            key, (num_queries, vocab_size), dtype=dtype
        )

    # Run Pallas implementation
    result = tax.top_k(logits, k=k, block_token=8, interpret=False)

    # Validate results using verify_topk_output
    validation = verify_topk_output(logits, result)

    assert bool(validation.all()), (
        f"Top-k validation failed for {dtype}: {int(validation.sum())}/{num_queries} rows passed"
    )

