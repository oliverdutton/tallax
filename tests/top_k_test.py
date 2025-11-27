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
    """Test top_k with different configurations."""
    num_queries = 16
    vocab_size = 201088
    k = 64
    # Generate test data
    key = jax.random.key(0)
    logits = jax.random.normal(
        key, (num_queries, vocab_size), dtype=jnp.float32
    ).astype(jnp.bfloat16)

    # Run Pallas implementation
    pallas_result = tax.top_k(logits, k=k, block_size=8)

    # Validate results
    validation = check_topk_out(logits, pallas_result)
    assert validation.all(), f"Top-k validation failed: {validation.sum()}/{num_queries} rows passed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
