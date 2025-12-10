
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from tallax._src.gather import take_along_axis
from tallax._src.utils import is_cpu_platform

@pytest.mark.parametrize("num_tokens", [8, 16, 13])
@pytest.mark.parametrize("vocab_size", [128, 256, 300])
@pytest.mark.parametrize("k", [64, 128, 200, 300])
@pytest.mark.parametrize("axis", [0, 1])
def test_take_along_axis(num_tokens, vocab_size, k, axis):
    key = jax.random.PRNGKey(0)
    key_vals, key_idxs = jax.random.split(key)

    if axis == 1:
        # Original gather test case: values (num_tokens, vocab_size), indices (num_tokens, k)
        values = jax.random.normal(key_vals, (num_tokens, vocab_size))
        indices = jax.random.randint(key_idxs, (num_tokens, k), 0, vocab_size)
    else:  # axis == 0
        # Transposed case: values (vocab_size, num_tokens), indices (k, num_tokens)
        values = jax.random.normal(key_vals, (vocab_size, num_tokens))
        indices = jax.random.randint(key_idxs, (k, num_tokens), 0, vocab_size)

    # Expected result using jax.numpy
    expected = jnp.take_along_axis(values, indices, axis=axis)

    # Run Pallas take_along_axis
    result = take_along_axis(values, indices, axis=axis, interpret=is_cpu_platform())

    np.testing.assert_array_equal(result, expected)

def test_take_along_axis_large_k():
    # Explicitly test (8, 1024) as requested in review
    num_tokens = 8
    vocab_size = 2048
    k = 1024

    key = jax.random.PRNGKey(1)
    values = jax.random.normal(key, (num_tokens, vocab_size))
    indices = jax.random.randint(key, (num_tokens, k), 0, vocab_size)

    expected = jnp.take_along_axis(values, indices, axis=1)
    result = take_along_axis(values, indices, axis=1, interpret=is_cpu_platform())

    np.testing.assert_array_equal(result, expected)
