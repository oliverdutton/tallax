
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tallax.gather import gather_pallas


@pytest.mark.parametrize("dim0, dim1", [(8, 128), (8, 256), (8, 512)])
def test_gather(dim0: int, dim1: int):
  # Generate random values and indices
  values = jax.random.normal(jax.random.PRNGKey(0), (dim0, dim1))
  indices = jax.random.randint(jax.random.PRNGKey(1), (dim0, 2), 0, dim1)

  # Compute gather using the Pallas implementation
  pallas_result = gather_pallas(values, indices)

  # Compute gather using the reference implementation
  reference_result = jnp.take_along_axis(values, indices, axis=1)

  # Check that the results are equal
  assert jnp.allclose(pallas_result, reference_result)
