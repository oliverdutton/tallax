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
def test_top_k():
    """Test top_k Pallas implementation."""
    num_queries = 16
    vocab_size = 201088
    k = 64

    # Generate test data
    key = jax.random.key(0)
    logits = jax.random.normal(
        key, (num_queries, vocab_size), dtype=jnp.float32
    ).astype(jnp.bfloat16)

    # Run Pallas implementation
    result = tax.top_k(logits, k=k, block_size=8, interpret=False)

    # Validate results using verify_topk_output
    validation = verify_topk_output(logits, result)

    if not validation.all():
        num_passed = validation.sum()
        pytest.fail(
            f"Top-k validation failed: {num_passed}/{num_queries} rows passed"
        )

