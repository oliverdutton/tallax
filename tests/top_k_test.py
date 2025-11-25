import jax
import jax.numpy as jnp
import pytest

from tallax import tax
from tallax.utils import is_cpu_platform


@jax.vmap
def check_topk_out(x, outs):
    """Validate top-k outputs for correctness."""
    assert x.ndim == 1
    out_vals, out_indexs = outs
    x_sorted = jnp.sort(x, descending=True)

    k = len(out_vals)
    n = len(x)
    valid = True

    # actual values must match
    valid &= (out_vals == x_sorted[:k]).all()

    # indices map to values correctly
    valid &= (x[out_indexs] == out_vals).all()

    # indices are all in bounds and unique
    i = jnp.unique(out_indexs, size=k, fill_value=-1)
    valid &= ((i >= 0) & (i < n)).all()
    return valid


def test_top_k():
    """Test top_k with different configurations for CPU vs TPU."""
    is_cpu = is_cpu_platform()

    if is_cpu:
        # On CPU, Pallas interpret mode can be unstable, so just test XLA
        pytest.skip("Skipping Pallas test on CPU - use TPU for full testing")

    # TPU configuration
    num_queries = 16
    vocab_size = 201088
    k = 64

    # Generate test data
    key = jax.random.key(0)
    logits = jax.random.normal(
        key, (num_queries, vocab_size), dtype=jnp.float32
    ).astype(jnp.bfloat16)

    # Run both implementations
    pallas_result = tax.top_k(logits, k=k, block_size=8, interpret=False)

    # Validate results
    validation = check_topk_out(logits, pallas_result)

    # Assert all rows pass validation
    assert validation.all(), f"Top-k validation failed: {validation.sum()}/{num_queries} rows passed"

    # Also compare with XLA reference implementation
    xla_result = jax.lax.top_k(logits, k=k)

    # Values should match exactly
    values_match = (pallas_result[0] == xla_result[0]).all()
    assert values_match, "Top-k values don't match XLA reference implementation"

    # Indices should map to the same values (may differ in order for ties)
    indices_valid = (logits[jnp.arange(num_queries)[:, None], pallas_result[1]] == xla_result[0]).all()
    assert indices_valid, "Top-k indices don't map to correct values"


def test_top_k_small():
    """Quick smoke test that validates the check_topk_out function with XLA."""
    k = 8
    num_queries = 2
    vocab_size = 128

    key = jax.random.key(42)
    logits = jax.random.normal(
        key, (num_queries, vocab_size), dtype=jnp.float32
    ).astype(jnp.bfloat16)

    # Use XLA top_k (reference implementation) to test validation function
    result = jax.lax.top_k(logits, k=k)

    # Validate - this tests that our validation function works correctly
    validation = check_topk_out(logits, result)
    assert validation.all(), "Validation function failed on XLA top-k output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
