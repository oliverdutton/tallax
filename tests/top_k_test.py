import jax
import jax.numpy as jnp
import pytest

from tallax import tax
from tallax.utils import is_cpu_platform, NUM_LANES
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

    assert bool(validation.all()), (
        f"Top-k validation failed: {int(validation.sum())}/{num_queries} rows passed"
    )


@pytest.mark.skipif(
    is_cpu_platform(),
    reason="Bitonic top-k tests require TPU/GPU - CPU uses interpret mode which is slow"
)
def test_bitonic_topk():
    """Test bitonic top_k implementation for k=128."""
    num_tokens = 32
    vocab_size = 8192
    k = NUM_LANES  # Must be 128

    # Generate test data
    key = jax.random.key(42)
    logits = jax.random.normal(
        key, (num_tokens, vocab_size), dtype=jnp.float32
    ).astype(jnp.bfloat16)

    # Run bitonic implementation
    result = tax.top_k(logits, k=k, bitonic=True, interpret=False)

    # Validate results using verify_topk_output
    validation = verify_topk_output(logits, result)

    assert bool(validation.all()), (
        f"Bitonic top-k validation failed: {int(validation.sum())}/{num_tokens} rows passed"
    )


def test_bitonic_topk_error_handling():
    """Test bitonic top_k error handling."""
    num_tokens = 32
    vocab_size = 8192

    key = jax.random.key(0)
    logits = jax.random.normal(key, (num_tokens, vocab_size))

    # Test k != 128
    with pytest.raises(ValueError, match="only supports k=NUM_LANES"):
        tax.top_k(logits, k=64, bitonic=True)

    # Test vocab_size not multiple of 128
    bad_logits = jax.random.normal(key, (num_tokens, 8000))
    with pytest.raises(ValueError, match="must be multiple of NUM_LANES"):
        tax.top_k(bad_logits, k=128, bitonic=True)

    # Test num_tokens > 128
    large_logits = jax.random.normal(key, (256, 8192))
    with pytest.raises(ValueError, match="must be <= NUM_LANES"):
        tax.top_k(large_logits, k=128, bitonic=True)

