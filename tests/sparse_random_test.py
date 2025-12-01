import pytest
import jax
import jax.numpy as jnp
from tallax.tax.sparse_random import sparse_random_categorical, sparse_random_uniform

def test_sparse_random_categorical():
    key = jax.random.PRNGKey(0)
    key_ref = jnp.array([key], dtype=jnp.uint32)

    # Simple logits where one value is much larger to make it deterministic-ish
    # or just check that it runs and output is valid index.
    batch = 4
    vocab = 10
    logits = jnp.zeros((batch, vocab), dtype=jnp.float32)

    rows = jnp.repeat(jnp.arange(batch)[:, None], vocab, axis=1)
    cols = jnp.repeat(jnp.arange(vocab)[None, :], batch, axis=0)
    indices = (rows, cols)
    dim1_size = vocab

    out = sparse_random_categorical(key_ref, logits, indices, dim1_size)

    assert out.shape == (batch,)
    assert jnp.all(out >= 0)
    assert jnp.all(out < vocab)

def test_sparse_random_uniform_shapes():
    key = jax.random.PRNGKey(42)
    key_ref = jnp.array([key], dtype=jnp.uint32)

    indices = (jnp.array([0, 1]), jnp.array([0, 1]))
    dim1_size = 10

    u = sparse_random_uniform(key_ref, indices, dim1_size)
    assert u.shape == (2,)
    assert jnp.all(u >= 0)
    assert jnp.all(u < 1.0)

def test_sparse_random_uniform_key_shape_check():
    key = jax.random.PRNGKey(42) # shape (2,)
    indices = (jnp.array([0]), jnp.array([0]))
    dim1_size = 10

    with pytest.raises(ValueError, match="key_ref must be 2D"):
        sparse_random_uniform(key, indices, dim1_size)

def test_sparse_random_indices_check():
    key_ref = jnp.array([jax.random.PRNGKey(0)], dtype=jnp.uint32)
    indices = (jnp.array([0]),) # length 1
    dim1_size = 10

    with pytest.raises(ValueError, match="indices must be length 2"):
        sparse_random_uniform(key_ref, indices, dim1_size)
