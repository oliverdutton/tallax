
import pytest
import jax
import jax.numpy as jnp
from tallax import tax
from tallax.utils import is_cpu_platform

def test_cumsum_correctness():
    shape = (8, 128)
    key = jax.random.key(42)
    # Use float to check for precision/logic, but int is good for exactness
    x = jax.random.randint(key, shape, 0, 100, dtype=jnp.int32)

    # Reference
    expected = jnp.cumsum(x, axis=1)

    # Test with different m values
    # m=0: all steps use mask/roll
    # m=128: all steps use permute
    # m=4: steps 1,2 permute; 4,8,16,32,64 mask
    # m=64: steps 1..32 permute; 64 mask
    ms = [0, 1, 4, 32, 64, 128]

    interpret = is_cpu_platform()

    for m in ms:
        actual = tax.cumsum(x, m=m, interpret=interpret)

        assert jnp.array_equal(actual, expected), f"Failed for m={m}"

if __name__ == "__main__":
    test_cumsum_correctness()
