
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from tallax import tax
from tallax.utils import is_cpu_platform

def test_cumsum_correctness():
    shape = (8, 128)
    key = jax.random.key(42)
    # Use float to check for precision/logic, but int is good for exactness
    x = jax.random.randint(key, shape, 0, 100, dtype=jnp.int32)

    # Reference
    expected = jnp.cumsum(x, axis=1)

    interpret = is_cpu_platform()

    actual = tax.cumsum(x, axis=1, interpret=interpret)

    np.testing.assert_array_equal(actual, expected)

if __name__ == "__main__":
    test_cumsum_correctness()
