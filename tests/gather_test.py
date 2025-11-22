
import pytest
import jax
import jax.numpy as jnp
from tallax import tax

@pytest.mark.parametrize("num_tokens", [8, 16, 13])
@pytest.mark.parametrize("vocab_size", [128, 256, 300])
@pytest.mark.parametrize("k", [64, 128, 200, 300])
def test_gather_correctness(num_tokens, vocab_size, k):
    key = jax.random.PRNGKey(0)
    key_vals, key_idxs = jax.random.split(key)

    values = jax.random.normal(key_vals, (num_tokens, vocab_size))
    # random indices in range [0, vocab_size)
    indices = jax.random.randint(key_idxs, (num_tokens, k), 0, vocab_size)

    # Expected result using jax.numpy
    # indices is (N, K), values is (N, V). We want to gather values[i, indices[i, j]]
    expected = jax.vmap(lambda v, i: v[i])(values, indices)

    # Run Pallas gather
    result = tax.gather(values, indices, interpret=True)

    assert jnp.allclose(result, expected)

def test_gather_large_k_explicit():
    # Explicitly test (8, 1024) as requested in review
    num_tokens = 8
    vocab_size = 2048
    k = 1024

    key = jax.random.PRNGKey(1)
    values = jax.random.normal(key, (num_tokens, vocab_size))
    indices = jax.random.randint(key, (num_tokens, k), 0, vocab_size)

    expected = jax.vmap(lambda v, i: v[i])(values, indices)
    result = tax.gather(values, indices, interpret=True)

    assert jnp.allclose(result, expected)
