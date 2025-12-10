
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
def test_sparse_random_categorical_axis1(seed):
    """Test sparse_random_categorical axis=1 by comparing against masked dense array."""
    key = jax.random.key(seed)

    # Generate dense logits (16, 256)
    dense_shape = (16, 256)
    logits = jax.random.normal(jax.random.key(seed + 100), dense_shape)

    # For each row, randomly select 128 columns to keep
    mask_key = jax.random.key(seed + 200)
    # Generate 128 column indices for each row
    all_cols = jnp.tile(jnp.arange(dense_shape[1]), (dense_shape[0], 1))  # (16, 256)
    perm_cols = jax.vmap(lambda k, cols: jax.random.permutation(k, cols))(
        jax.random.split(mask_key, dense_shape[0]),
        all_cols
    )
    selected_cols = perm_cols[:, :128]  # (16, 128) - column indices to keep for each row

    # Create indices for sparse array
    indices_0 = jnp.tile(jnp.arange(dense_shape[0])[:, None], (1, 128))  # (16, 128)
    indices_1 = selected_cols  # (16, 128)

    # Create masked logits (set non-selected to -inf)
    masked_logits = jnp.full(dense_shape, -1e12)
    # For each row, set the selected column positions
    for row in range(dense_shape[0]):
        masked_logits = masked_logits.at[row, selected_cols[row]].set(logits[row, selected_cols[row]])

    # Sample from dense masked array
    dense_result = jax.random.categorical(key, masked_logits, axis=1)

    # Sample from sparse array
    sparse_logits = logits[indices_0, indices_1]  # (16, 128)
    sparse_result = sparse_random_categorical(
        key,
        sparse_logits,
        [indices_0, indices_1],
        dim1_size=dense_shape[1],
        axis=1
    )

    # sparse_result[1] contains the sparse column indices (0-127) for each row
    # Map these back to dense column indices (0-255)
    sparse_col_indices = sparse_result[1].squeeze()  # (16,)
    row_indices = jnp.arange(dense_shape[0])
    mapped_result = indices_1[row_indices, sparse_col_indices]

    # Should match exactly
    np.testing.assert_array_equal(mapped_result, dense_result,
        err_msg="sparse_random_categorical should match dense categorical for axis=1")


@pytest.mark.parametrize("seed", [111, 222, 333])
def test_sparse_random_categorical_axis0(seed):
    """Test sparse_random_categorical axis=0 by comparing against masked dense array."""
    key = jax.random.key(seed)

    # Generate dense logits (16, 256) - but we'll use power-of-2 for axis=0
    # Since axis=0 sampling requires dim0 to be power of 2, use (128, 256)
    dense_shape = (128, 256)
    logits = jax.random.normal(jax.random.key(seed + 100), dense_shape)

    # For each column, randomly select 128 rows to keep
    mask_key = jax.random.key(seed + 200)
    # Generate 128 row indices for each column
    all_rows = jnp.tile(jnp.arange(dense_shape[0]), (dense_shape[1], 1))  # (256, 128)
    perm_rows = jax.vmap(lambda k, rows: jax.random.permutation(k, rows))(
        jax.random.split(mask_key, dense_shape[1]),
        all_rows
    )
    selected_rows = perm_rows[:, :128].T  # (128, 256) - row indices to keep for each column

    # Create indices for sparse array
    indices_0 = selected_rows  # (128, 256)
    indices_1 = jnp.tile(jnp.arange(dense_shape[1]), (128, 1))  # (128, 256)

    # Create masked logits (set non-selected to -inf)
    masked_logits = jnp.full(dense_shape, -1e12)
    # For each column, set the selected row positions
    for col in range(dense_shape[1]):
        masked_logits = masked_logits.at[selected_rows[:, col], col].set(logits[selected_rows[:, col], col])

    # Sample from dense masked array
    dense_result = jax.random.categorical(key, masked_logits, axis=0)

    # Sample from sparse array
    sparse_logits = logits[indices_0, indices_1]  # (128, 256)
    sparse_result = sparse_random_categorical(
        key,
        sparse_logits,
        [indices_0, indices_1],
        dim1_size=dense_shape[1],
        axis=0
    )

    # sparse_result[0] contains the sparse row indices (0-127) for each column
    # Map these back to dense row indices (0-127)
    sparse_row_indices = sparse_result[0].squeeze()  # (256,)
    col_indices = jnp.arange(dense_shape[1])
    mapped_result = indices_0[sparse_row_indices, col_indices]

    # Should match exactly
    np.testing.assert_array_equal(mapped_result, dense_result,
        err_msg="sparse_random_categorical should match dense categorical for axis=0")


if __name__ == "__main__":
    print("Running sparse_random_uniform tests...")
    test_sparse_random_uniform(42, 0.0, 1.0)
    test_sparse_random_uniform(123, -1.0, 1.0)
    print("sparse_random_uniform tests passed!")

    print("\nRunning sparse_random_categorical tests...")
    test_sparse_random_categorical_axis1(789)
    test_sparse_random_categorical_axis0(111)
    print("sparse_random_categorical tests passed!")

    print("\nAll tests passed!")
