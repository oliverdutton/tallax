
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from tallax._src.sparse_random import sparse_random_uniform, sparse_random_categorical


@pytest.mark.parametrize("shape", [(8, 16), (16, 32), (32, 64)])
@pytest.mark.parametrize("minval,maxval", [(0.0, 1.0), (-1.0, 1.0), (5.0, 10.0)])
def test_sparse_random_uniform_matches_jax(shape, minval, maxval):
    """Test that sparse_random_uniform matches jax.random.uniform."""
    dim0_size, dim1_size = shape
    key = jax.random.key(42)

    # Create sparse indices using broadcasted_iota
    indices_0 = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    indices_1 = jax.lax.broadcasted_iota(jnp.int32, shape, 1)

    # Generate sparse random uniform
    result_sparse = sparse_random_uniform(
        key,
        [indices_0, indices_1],
        dim1_size,
        dtype=jnp.float32,
        minval=minval,
        maxval=maxval
    )

    # Generate standard JAX random uniform for comparison
    result_jax = jax.random.uniform(
        key,
        shape=shape,
        dtype=jnp.float32,
        minval=minval,
        maxval=maxval
    )

    # Check that results match
    np.testing.assert_allclose(result_sparse, result_jax, rtol=1e-6, atol=1e-6,
        err_msg="sparse_random_uniform should match jax.random.uniform")


@pytest.mark.parametrize("shape", [(8, 16), (16, 32), (32, 64)])
@pytest.mark.parametrize("axis", [0, -2])  # Only test axis=0 which matches JAX behavior
def test_sparse_random_categorical_matches_jax(shape, axis):
    """Test that sparse_random_categorical matches jax.random.categorical for axis=0."""
    dim0_size, dim1_size = shape
    key = jax.random.key(123)

    # Create sparse indices with same shape as logits
    indices_0 = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    indices_1 = jax.lax.broadcasted_iota(jnp.int32, shape, 1)

    # Create random 2D logits
    logits = jax.random.normal(jax.random.key(100), shape)

    # Generate sparse random categorical
    result_sparse = sparse_random_categorical(
        key,
        logits,
        [indices_0, indices_1],
        dim1_size,
        axis=axis
    )

    # Generate standard JAX random categorical for comparison
    result_jax_idx = jax.random.categorical(
        key,
        logits,
        axis=axis
    )

    # sparse_random_categorical returns indices as [idx0, idx1]
    # For axis=0, check the first index array (row indices)
    np.testing.assert_array_equal(result_sparse[0].squeeze(), result_jax_idx,
        err_msg="sparse_random_categorical should match jax.random.categorical for axis=0")


@pytest.mark.parametrize("shape", [(8, 16), (16, 32)])
@pytest.mark.parametrize("axis", [1, -1])  # Test axis=1 for completeness
def test_sparse_random_categorical_axis1(shape, axis):
    """Test that sparse_random_categorical works with axis=1."""
    dim0_size, dim1_size = shape
    key = jax.random.key(456)

    # Create sparse indices with same shape as logits
    indices_0 = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    indices_1 = jax.lax.broadcasted_iota(jnp.int32, shape, 1)

    # Create random 2D logits
    logits = jax.random.normal(jax.random.key(200), shape)

    # Generate sparse random categorical
    result_sparse = sparse_random_categorical(
        key,
        logits,
        [indices_0, indices_1],
        dim1_size,
        axis=axis
    )

    # Verify structure
    assert len(result_sparse) == 2, "Should return 2 index arrays"
    assert result_sparse[0].shape == result_sparse[1].shape, "Index arrays should have same shape"

    # For axis=1, verify that sampled indices are valid
    assert jnp.all(result_sparse[1].squeeze() >= 0), "Indices should be >= 0"
    assert jnp.all(result_sparse[1].squeeze() < dim1_size), \
        f"Indices should be < {dim1_size}"


if __name__ == "__main__":
    # Run basic tests
    print("Running sparse_random_uniform tests...")
    test_sparse_random_uniform_matches_jax((16, 32), 0.0, 1.0)
    print("sparse_random_uniform test passed!")

    print("\nRunning sparse_random_categorical tests...")
    test_sparse_random_categorical_matches_jax((8, 16), 0)
    test_sparse_random_categorical_axis1((8, 16), 1)
    print("sparse_random_categorical tests passed!")

    print("\nAll tests passed!")
