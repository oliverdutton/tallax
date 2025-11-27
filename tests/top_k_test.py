import jax
import jax.numpy as jnp
import pytest

from tallax import tax
from tallax.utils import is_cpu_platform
from tallax.test_utils import check_topk_out


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
    result = tax.top_k(logits, k=k, block_size=8, interpret=is_cpu_platform())

    # Validate results using check_topk_out
    validation = check_topk_out(logits, result)

    if not validation.all():
        num_passed = validation.sum()
        pytest.fail(
            f"Top-k validation failed: {num_passed}/{num_queries} rows passed"
        )

