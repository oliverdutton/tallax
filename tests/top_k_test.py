import jax
import jax.numpy as jnp
import pytest

from tallax import tax
from tallax._src.utils import is_cpu_platform
from tallax._src.test_utils import verify_topk_output


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

    # Validate results using verify_topk_output (axis=1 is default)
    validation = verify_topk_output(logits, result, axis=1)

    assert bool(validation.all()), (
        f"Top-k validation failed: {int(validation.sum())}/{num_queries} rows passed"
    )

