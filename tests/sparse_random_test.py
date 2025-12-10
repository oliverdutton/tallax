
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

    # Generate dense logits
    dense_shape = (16, 256)
    logits = jax.random.normal(jax.random.key(seed + 100), dense_shape)

    # Create mask with 128 True values per row (axis=1) or column (axis=0)
    mask_key = jax.random.key(seed + 200)

    if axis == 1:
        # Sample along axis=1: for each row, select 128 columns
        sparse_shape = (16, 128)
        # For each row, randomly select 128 column indices
        indices_0 = jnp.tile(jnp.arange(dense_shape[0])[:, None], (1, 128))  # (16, 128)
        # Generate random column indices for each row
        all_cols = jnp.tile(jnp.arange(dense_shape[1]), (dense_shape[0], 1))  # (16, 256)
        perm_cols = jax.vmap(lambda k, cols: jax.random.permutation(k, cols))(
            jax.random.split(mask_key, dense_shape[0]),
            all_cols
        )
        indices_1 = perm_cols[:, :128]  # (16, 128)
    else:  # axis == 0
        # Sample along axis=0: for each column, select 128 rows
        sparse_shape = (128, 256)
        # For each column, randomly select 128 row indices
        all_rows = jnp.tile(jnp.arange(dense_shape[0]), (dense_shape[1], 1))  # (256, 16)
        perm_rows = jax.vmap(lambda k, rows: jax.random.permutation(k, rows))(
            jax.random.split(mask_key, dense_shape[1]),
            all_rows
        )
        indices_0 = perm_rows[:, :128].T  # (128, 256)
        indices_1 = jnp.tile(jnp.arange(dense_shape[1]), (128, 1))  # (128, 256)

    # Create masked logits (set non-selected positions to -inf)
    masked_logits = jnp.full(dense_shape, -1e12)
    masked_logits = masked_logits.at[indices_0, indices_1].set(logits[indices_0, indices_1])

    # Sample from dense masked array
    dense_result = jax.random.categorical(key, masked_logits, axis=axis)

    # Sample from sparse array
    sparse_logits = logits[indices_0, indices_1]
    sparse_result = sparse_random_categorical(
        key,
        sparse_logits,
        [indices_0, indices_1],
        dim1_size=dense_shape[1],
        axis=axis
    )

    # Extract the relevant index array and map back to dense indices
    if axis == 0:
        # For axis=0, result is which row was selected for each column
        # sparse_result[0] contains the sparse row indices (0-127)
        # We need to map these to dense row indices (0-15)
        sparse_row_indices = sparse_result[0].squeeze()  # Shape: (256,)
        # For each column, get the dense row index
        col_indices = jnp.arange(dense_shape[1])
        mapped_result = indices_0[sparse_row_indices, col_indices]
    else:  # axis == 1
        # For axis=1, result is which column was selected for each row
        # sparse_result[1] contains the sparse column indices (0-127)
        # We need to map these to dense column indices (0-255)
        sparse_col_indices = sparse_result[1].squeeze()  # Shape: (16,)
        # For each row, get the dense column index
        row_indices = jnp.arange(dense_shape[0])
        mapped_result = indices_1[row_indices, sparse_col_indices]

    # Should match exactly (categorical returns int indices)
    np.testing.assert_array_equal(mapped_result, dense_result,
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
