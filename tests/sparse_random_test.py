
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from tallax._src.sparse_random import sparse_random_uniform, sparse_random_categorical


@pytest.mark.parametrize("seed", [42, 123, 456])
@pytest.mark.parametrize("minval,maxval", [(0.0, 1.0), (-1.0, 1.0), (5.0, 10.0)])
def test_sparse_random_uniform(seed, minval, maxval):
    """Test sparse_random_uniform by comparing against indexed dense array."""
    key = jax.random.key(seed)

    # Generate dense random array
    dense_shape = (16, 256)
    dense_uniform = jax.random.uniform(
        key,
        shape=dense_shape,
        dtype=jnp.float32,
        minval=minval,
        maxval=maxval
    )

    # Generate random sparse indices
    sparse_shape = (8, 128)
    idx_key = jax.random.key(seed + 1000)
    indices_0 = jax.random.randint(idx_key, sparse_shape, 0, dense_shape[0])
    indices_1 = jax.random.randint(jax.random.key(seed + 2000), sparse_shape, 0, dense_shape[1])

    # Generate sparse random values
    sparse_uniform = sparse_random_uniform(
        key,
        [indices_0, indices_1],
        dim1_size=dense_shape[1],
        dtype=jnp.float32,
        minval=minval,
        maxval=maxval
    )

    # Index into dense array at the sparse positions
    expected = dense_uniform[indices_0, indices_1]

    # Should match exactly (or within FP32 epsilon for scaled ranges)
    if minval == 0.0 and maxval == 1.0:
        # For [0, 1) range, should be exact
        np.testing.assert_array_equal(sparse_uniform, expected,
            err_msg="sparse_random_uniform should match indexed dense array exactly")
    else:
        # For scaled ranges, allow 1 ULP difference due to FP arithmetic
        np.testing.assert_allclose(sparse_uniform, expected, rtol=0, atol=1e-6,
            err_msg="sparse_random_uniform should match indexed dense array")


@pytest.mark.parametrize("seed", [789, 321, 654])
@pytest.mark.parametrize("axis", [0, 1])
def test_sparse_random_categorical(seed, axis):
    """Test sparse_random_categorical by comparing against masked dense array."""
    key = jax.random.key(seed)

    # Dense shape and sparse shape
    dense_shape = (16, 256)
    sparse_shape = (8, 128)

    # Generate SPARSE logits (not extracted from dense)
    logits_key = jax.random.key(seed + 100)
    sparse_logits = jax.random.normal(logits_key, sparse_shape)

    # Create sparse indices
    if axis == 1:
        # Sample along axis=1: for each of 8 rows, select 128 columns
        # indices_0: broadcasted iota for rows
        indices_0 = jax.lax.broadcasted_iota(jnp.int32, sparse_shape, 0)
        # indices_1: random permutation of columns for each row
        mask_key = jax.random.key(seed + 200)
        all_cols = jnp.tile(jnp.arange(dense_shape[1]), (sparse_shape[0], 1))
        perm_cols = jax.vmap(lambda k, cols: jax.random.permutation(k, cols))(
            jax.random.split(mask_key, sparse_shape[0]),
            all_cols
        )
        indices_1 = perm_cols[:, :sparse_shape[1]]
    else:  # axis == 0
        # Sample along axis=0: for each of 128 columns, select 8 rows
        # indices_1: broadcasted iota for columns
        indices_1 = jax.lax.broadcasted_iota(jnp.int32, sparse_shape, 1)
        # indices_0: random permutation of rows for each column
        mask_key = jax.random.key(seed + 200)
        all_rows = jnp.tile(jnp.arange(dense_shape[0]), (sparse_shape[1], 1))
        perm_rows = jax.vmap(lambda k, rows: jax.random.permutation(k, rows))(
            jax.random.split(mask_key, sparse_shape[1]),
            all_rows
        )
        indices_0 = perm_rows[:, :sparse_shape[0]].T

    # Create dense masked array: all -1e12 except at sparse indices
    dense_masked = jnp.full(dense_shape, -1e12)
    dense_masked = dense_masked.at[indices_0, indices_1].set(sparse_logits)

    # Sample from dense masked array using jax.random.categorical
    dense_result = jax.random.categorical(key, dense_masked, axis=axis)

    # Sample from sparse array
    sparse_result = sparse_random_categorical(
        key,
        sparse_logits,
        [indices_0, indices_1],
        dim1_size=dense_shape[1],
        axis=axis
    )

    # Map sparse indices back to dense indices and compare
    # sparse_result contains the VALUES of indices at argmax positions (not positions themselves)
    if axis == 0:
        # Sampling along axis 0: for each column, select a row
        # sparse_result[0] contains the DENSE row indices (from indices_0 array)
        # indices_1 is broadcasted iota, so we compare for first 128 columns
        mapped_result = sparse_result[0].squeeze()
        expected_result = dense_result[:128]
    else:  # axis == 1
        # Sampling along axis 1: for each row, select a column
        # sparse_result[1] contains the DENSE column indices (from indices_1 array)
        # indices_0 is broadcasted iota, so we compare for first 8 rows
        mapped_result = sparse_result[1].squeeze()
        expected_result = dense_result[:8]

    # Should match exactly (categorical returns int indices)
    np.testing.assert_array_equal(mapped_result, expected_result,
        err_msg=f"sparse_random_categorical should match dense categorical for axis={axis}")


if __name__ == "__main__":
    print("Running sparse_random_uniform tests...")
    test_sparse_random_uniform(42, 0.0, 1.0)
    test_sparse_random_uniform(123, -1.0, 1.0)
    print("sparse_random_uniform tests passed!")

    print("\nRunning sparse_random_categorical tests...")
    test_sparse_random_categorical(789, 0)
    test_sparse_random_categorical(321, 1)
    print("sparse_random_categorical tests passed!")

    print("\nAll tests passed!")
