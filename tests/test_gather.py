
import pytest
import jax
import jax.numpy as jnp
from tallax.tax.gather import gather

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
    result = gather(values, indices, interpret=True)

    assert jnp.allclose(result, expected)

def test_gather_large_k():
    # k larger than multiple tiles
    num_tokens = 32
    vocab_size = 1024
    k = 512

    key = jax.random.PRNGKey(1)
    values = jax.random.normal(key, (num_tokens, vocab_size))
    indices = jax.random.randint(key, (num_tokens, k), 0, vocab_size)

    expected = jax.vmap(lambda v, i: v[i])(values, indices)
    result = gather(values, indices, interpret=True)

    assert jnp.allclose(result, expected)
