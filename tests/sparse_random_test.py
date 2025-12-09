
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from tallax._src.sparse_random import sparse_random_uniform, sparse_random_categorical


@pytest.mark.parametrize("shape", [(8, 16), (16, 32), (32, 64), (13, 27)])
@pytest.mark.parametrize("dtype", [jnp.float32])
def test_sparse_random_uniform_basic(shape, dtype):
    """Test basic functionality of sparse_random_uniform."""
    dim0_size, dim1_size = shape
    key = jax.random.key(42)
    key_ref = jnp.reshape(jax.random.key_data(key), (1, 2))

    # Create sparse indices using broadcasted_iota
    # indices_0 should vary along dimension 0 (rows)
    # indices_1 should vary along dimension 1 (columns)
    indices_0 = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    indices_1 = jax.lax.broadcasted_iota(jnp.int32, shape, 1)

    # Generate uniform random values
    result = sparse_random_uniform(
        key_ref,
        [indices_0, indices_1],
        dim1_size,
        dtype=dtype
    )

    # Check output shape
    assert result.shape == (dim0_size, dim1_size), \
        f"Expected shape {(dim0_size, dim1_size)}, got {result.shape}"

    # Check dtype
    assert result.dtype == dtype, \
        f"Expected dtype {dtype}, got {result.dtype}"

    # Check values are in [0, 1)
    assert jnp.all(result >= 0.0), "Values should be >= 0.0"
    assert jnp.all(result < 1.0), "Values should be < 1.0"


@pytest.mark.parametrize("minval,maxval", [(0.0, 1.0), (-1.0, 1.0), (5.0, 10.0), (-10.0, -5.0)])
def test_sparse_random_uniform_range(minval, maxval):
    """Test that sparse_random_uniform respects min/max values."""
    dim0_size, dim1_size = 16, 32
    key = jax.random.key(123)
    key_ref = jnp.reshape(jax.random.key_data(key), (1, 2))

    shape = (dim0_size, dim1_size)
    indices_0 = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    indices_1 = jax.lax.broadcasted_iota(jnp.int32, shape, 1)

    result = sparse_random_uniform(
        key_ref,
        [indices_0, indices_1],
        dim1_size,
        dtype=jnp.float32,
        minval=minval,
        maxval=maxval
    )

    # Check values are in [minval, maxval)
    assert jnp.all(result >= minval), \
        f"Values should be >= {minval}, got min {jnp.min(result)}"
    assert jnp.all(result < maxval), \
        f"Values should be < {maxval}, got max {jnp.max(result)}"


def test_sparse_random_uniform_deterministic():
    """Test that same key produces same results."""
    dim0_size, dim1_size = 16, 32
    key = jax.random.key(456)
    key_ref = jnp.reshape(jax.random.key_data(key), (1, 2))

    shape = (dim0_size, dim1_size)
    indices_0 = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    indices_1 = jax.lax.broadcasted_iota(jnp.int32, shape, 1)

    result1 = sparse_random_uniform(
        key_ref,
        [indices_0, indices_1],
        dim1_size
    )

    result2 = sparse_random_uniform(
        key_ref,
        [indices_0, indices_1],
        dim1_size
    )

    np.testing.assert_array_equal(result1, result2,
        "Same key should produce same results")


def test_sparse_random_uniform_different_keys():
    """Test that different keys produce different results."""
    dim0_size, dim1_size = 16, 32
    key1 = jax.random.key(789)
    key2 = jax.random.key(790)
    key_ref1 = jnp.reshape(jax.random.key_data(key1), (1, 2))
    key_ref2 = jnp.reshape(jax.random.key_data(key2), (1, 2))

    shape = (dim0_size, dim1_size)
    indices_0 = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    indices_1 = jax.lax.broadcasted_iota(jnp.int32, shape, 1)

    result1 = sparse_random_uniform(
        key_ref1,
        [indices_0, indices_1],
        dim1_size
    )

    result2 = sparse_random_uniform(
        key_ref2,
        [indices_0, indices_1],
        dim1_size
    )

    # Results should be different
    assert not jnp.allclose(result1, result2), \
        "Different keys should produce different results"


def test_sparse_random_uniform_distribution():
    """Test that sparse_random_uniform produces roughly uniform distribution."""
    dim0_size, dim1_size = 128, 256
    key = jax.random.key(999)
    key_ref = jnp.reshape(jax.random.key_data(key), (1, 2))

    shape = (dim0_size, dim1_size)
    indices_0 = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    indices_1 = jax.lax.broadcasted_iota(jnp.int32, shape, 1)

    result = sparse_random_uniform(
        key_ref,
        [indices_0, indices_1],
        dim1_size
    )

    # Check mean is approximately 0.5
    mean = jnp.mean(result)
    assert 0.45 < mean < 0.55, \
        f"Mean should be approximately 0.5, got {mean}"

    # Check standard deviation is approximately sqrt(1/12) ≈ 0.289
    std = jnp.std(result)
    expected_std = jnp.sqrt(1.0 / 12.0)
    assert 0.25 < std < 0.33, \
        f"Std should be approximately {expected_std:.3f}, got {std}"


@pytest.mark.parametrize("shape", [(8, 16), (16, 32), (32, 64), (64, 27)])
def test_sparse_random_categorical_basic(shape):
    """Test basic functionality of sparse_random_categorical."""
    dim0_size, dim1_size = shape
    key = jax.random.key(42)
    key_ref = jnp.reshape(jax.random.key_data(key), (1, 2))

    # Create sparse indices with same shape as logits
    indices_0 = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    indices_1 = jax.lax.broadcasted_iota(jnp.int32, shape, 1)

    # Create random 2D logits
    logits = jax.random.normal(jax.random.key(100), shape)

    # Generate categorical samples (axis must be 0)
    sampled_indices = sparse_random_categorical(
        key_ref,
        logits,
        [indices_0, indices_1],
        dim1_size,
        axis=0
    )

    # Check output shape - should return indices for each dimension
    assert len(sampled_indices) == 2, \
        f"Expected 2 index arrays, got {len(sampled_indices)}"

    # Both index arrays should have the same shape
    assert sampled_indices[0].shape == sampled_indices[1].shape, \
        "Index arrays should have same shape"

    # Check that sampled indices are valid
    assert jnp.all(sampled_indices[0] >= 0), "Indices should be >= 0"
    assert jnp.all(sampled_indices[0] < dim0_size), \
        f"Indices should be < {dim0_size}"


@pytest.mark.parametrize("shape", [(8, 16), (16, 32), (32, 64)])
def test_sparse_random_categorical_deterministic(shape):
    """Test that same key produces same categorical samples."""
    dim0_size, dim1_size = shape
    key = jax.random.key(456)
    key_ref = jnp.reshape(jax.random.key_data(key), (1, 2))

    indices_0 = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    indices_1 = jax.lax.broadcasted_iota(jnp.int32, shape, 1)

    logits = jax.random.normal(jax.random.key(200), shape)

    result1 = sparse_random_categorical(
        key_ref,
        logits,
        [indices_0, indices_1],
        dim1_size,
        axis=0
    )

    result2 = sparse_random_categorical(
        key_ref,
        logits,
        [indices_0, indices_1],
        dim1_size,
        axis=0
    )

    # Both index arrays should match
    for i in range(len(result1)):
        np.testing.assert_array_equal(result1[i], result2[i],
            f"Same key should produce same results for index array {i}")


def test_sparse_random_categorical_different_keys():
    """Test that different keys produce different categorical samples."""
    dim0_size, dim1_size = 8, 16
    key1 = jax.random.key(789)
    key2 = jax.random.key(790)
    key_ref1 = jnp.reshape(jax.random.key_data(key1), (1, 2))
    key_ref2 = jnp.reshape(jax.random.key_data(key2), (1, 2))

    shape = (dim0_size, dim1_size)
    indices_0 = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    indices_1 = jax.lax.broadcasted_iota(jnp.int32, shape, 1)

    logits = jax.random.normal(jax.random.key(300), shape)

    result1 = sparse_random_categorical(
        key_ref1,
        logits,
        [indices_0, indices_1],
        dim1_size,
        axis=0
    )

    result2 = sparse_random_categorical(
        key_ref2,
        logits,
        [indices_0, indices_1],
        dim1_size,
        axis=0
    )

    # Results should be different (at least one index array should differ)
    results_differ = False
    for i in range(len(result1)):
        if not jnp.array_equal(result1[i], result2[i]):
            results_differ = True
            break

    assert results_differ, \
        "Different keys should produce different results"


def test_sparse_random_categorical_biased_logits():
    """Test that categorical sampling respects logit probabilities."""
    dim0_size, dim1_size = 8, 128

    # Create logits with one row heavily favored along axis 0
    logits = jnp.ones((dim0_size, dim1_size)) * (-10.0)
    favored_idx = 3
    logits = logits.at[favored_idx, :].set(10.0)

    # Sample and check that the function runs without error
    key = jax.random.key(1000)
    key_ref = jnp.reshape(jax.random.key_data(key), (1, 2))

    shape = (dim0_size, dim1_size)
    indices_0 = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    indices_1 = jax.lax.broadcasted_iota(jnp.int32, shape, 1)

    result = sparse_random_categorical(
        key_ref,
        logits,
        [indices_0, indices_1],
        dim1_size,
        axis=0
    )

    # Verify result structure
    assert len(result) == 2, "Should return 2 index arrays"
    # With heavily biased logits along axis 0, most samples should favor the biased index
    # The result[0] contains the selected indices along axis 0
    # Since favored_idx has much higher logits, it should be selected frequently
    assert jnp.sum(result[0] == favored_idx) > dim1_size * 0.8, \
        "Biased index should be selected frequently"


def test_sparse_random_categorical_uniform_logits():
    """Test categorical sampling with uniform logits."""
    dim0_size, dim1_size = 8, 256
    key = jax.random.key(2000)
    key_ref = jnp.reshape(jax.random.key_data(key), (1, 2))

    # Create uniform logits (all equal)
    shape = (dim0_size, dim1_size)
    logits = jnp.zeros(shape)

    indices_0 = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    indices_1 = jax.lax.broadcasted_iota(jnp.int32, shape, 1)

    result = sparse_random_categorical(
        key_ref,
        logits,
        [indices_0, indices_1],
        dim1_size,
        axis=0
    )

    # Check that result is valid
    assert len(result) == 2, "Should return 2 index arrays"
    assert result[0].shape == result[1].shape, "Index arrays should have same shape"


if __name__ == "__main__":
    # Run basic tests
    print("Running sparse_random_uniform tests...")
    test_sparse_random_uniform_basic((16, 32), jnp.float32)
    test_sparse_random_uniform_range(0.0, 1.0)
    test_sparse_random_uniform_deterministic()
    test_sparse_random_uniform_different_keys()
    test_sparse_random_uniform_distribution()

    print("\nRunning sparse_random_categorical tests...")
    test_sparse_random_categorical_basic((8, 16))
    test_sparse_random_categorical_deterministic((8, 16))
    test_sparse_random_categorical_different_keys()
    test_sparse_random_categorical_biased_logits()
    test_sparse_random_categorical_uniform_logits()

    print("\nAll tests passed!")
