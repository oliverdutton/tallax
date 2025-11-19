
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tallax.gather import gather_pallas


@pytest.mark.parametrize("dim1", [128, 256, 512])
def test_gather(dim1: int):
  # Generate random values and indices
  values = jax.random.normal(jax.random.PRNGKey(0), (8, dim1))
  indices = jax.random.randint(jax.random.PRNGKey(1), (8, 2), 0, dim1)

  # Compute gather using the Pallas implementation
  pallas_result = gather_pallas(values, indices)

  # Compute gather using the reference implementation
  reference_result = jnp.take_along_axis(values, indices, axis=1)

  # Check that the results are equal
  assert jnp.allclose(pallas_result, reference_result)


@pytest.mark.parametrize("vmem_limit_bytes", [1024, 2048, 4096])
def test_gather_vmem_limit(vmem_limit_bytes: int):
  # Generate random values and indices
  values = jax.random.normal(jax.random.PRNGKey(0), (8, 512))
  indices = jax.random.randint(jax.random.PRNGKey(1), (8, 2), 0, 512)

  # Compute gather using the Pallas implementation with a VMEM limit
  pallas_result = gather_pallas(values, indices, vmem_limit_bytes=vmem_limit_bytes)

  # Compute gather using the reference implementation
  reference_result = jnp.take_along_axis(values, indices, axis=1)

  # Check that the results are equal
  assert jnp.allclose(pallas_result, reference_result)
