import functools
import pytest
import jax
import jax.numpy as jnp
from tallax import tax
from tallax._src.utils import is_cpu_platform


def verify_approx_topk_output(x, outs, k, axis=-1, recall_target=0.95):
    """Validate approximate top-k outputs for correctness.

    Similar to verify_topk_output but for approximate top-k:
    - Checks that >recall_target fraction of values are >= k-th threshold
    - Validates indices are unique and map to values correctly
    - Does not require exact match with true top-k

    Args:
        x: Input array (must be 2D)
        outs: Tuple of (values, indices) from approx_max_k (both must be 2D)
        k: Number of top elements requested
        axis: Axis along which top-k was computed (default -1)
        recall_target: Minimum fraction of correct top-k values (default 0.95)

    Returns:
        Boolean array indicating if the approx top-k output is valid for each batch element

    Raises:
        ValueError: If x or outputs are not 2D
    """
    if x.ndim != 2:
        raise ValueError(f"verify_approx_topk_output only supports 2D inputs, got {x.ndim}D")

    out_vals, out_indexs = outs

    if out_vals.ndim != 2 or out_indexs.ndim != 2:
        raise ValueError(f"verify_approx_topk_output requires 2D outputs, got values.ndim={out_vals.ndim}, indices.ndim={out_indexs.ndim}")

    # Normalize axis to 0 or 1
    axis = axis % 2
    batch_axis = 1 - axis

    @functools.partial(jax.vmap, in_axes=batch_axis)
    def verify_slice(x_slice, vals_slice, idxs_slice):
        """Verify a single slice."""
        # Get true top-k threshold using jax.lax.top_k
        true_topk_vals = jax.lax.top_k(x_slice, k)[0]
        threshold = true_topk_vals[-1]

        n = len(x_slice)
        valid = True

        # Check >recall_target fraction of values are >= threshold
        above_threshold = vals_slice >= threshold
        recall = jnp.mean(above_threshold)
        valid &= recall > recall_target

        # indices map to values correctly
        valid &= (x_slice[idxs_slice] == vals_slice).all()

        # indices are all in bounds and unique
        i = jnp.unique(idxs_slice, size=k, fill_value=-1)
        valid &= ((i >= 0) & (i < n)).all()
        return valid

    return verify_slice(x, out_vals, out_indexs)


@pytest.mark.parametrize("shape", [(8, 128), (16, 256), (32, 512), (64, 1024)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32])
@pytest.mark.parametrize("k", [10, 32, 64])
@pytest.mark.skipif(
    is_cpu_platform(),
    reason="approx_max_k tests require TPU/GPU - CPU uses interpret mode which is slow"
)
def test_approx_max_k_2d(shape, dtype, k):
    """Test approx_max_k with 2D inputs and axis=-1."""
    if k >= shape[1]:
        pytest.skip(f"k={k} >= vocab_size={shape[1]}")

    # Generate test data
    key = jax.random.key(42)
    if dtype == jnp.float32:
        operand = jax.random.normal(key, shape, dtype=dtype)
    else:
        operand = jax.random.randint(key, shape, 0, 1000, dtype=dtype)

    # Run approx_max_k
    outputs = tax.approx_max_k(operand, k=k, reduction_dimension=-1)

    # Validate results
    validation = verify_approx_topk_output(operand, outputs, k=k, axis=-1)

    assert validation.all(), (
        f"approx_max_k validation failed for shape {shape}, dtype {dtype}, k={k}: "
        f"{int(validation.sum())}/{shape[0]} rows passed"
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32])
@pytest.mark.parametrize("k", [32, 64, 128])
@pytest.mark.skipif(
    is_cpu_platform(),
    reason="approx_max_k tests require TPU/GPU - CPU uses interpret mode which is slow"
)
def test_approx_max_k_large_vocab(dtype, k):
    """Test approx_max_k with large vocabulary (128, 8192) using iota pattern."""
    shape = (128, 8192)

    # Create iota pattern and reshape
    if dtype == jnp.float32:
        operand = jnp.arange(shape[0] * shape[1], dtype=dtype).reshape(shape)
        # Add some randomness to make it more realistic
        key = jax.random.key(123)
        operand = operand + jax.random.normal(key, shape, dtype=dtype) * 0.1
    else:
        operand = jnp.arange(shape[0] * shape[1], dtype=dtype).reshape(shape)

    # Run approx_max_k
    outputs = tax.approx_max_k(operand, k=k, reduction_dimension=-1)
    out_vals, out_indexs = outputs

    # Get true top-k threshold for each row using jax.lax.top_k
    true_topk = jax.lax.top_k(operand, k)
    thresholds = true_topk[0][:, -1]  # Last value in each row's top-k

    # Validate outputs
    batch_size = shape[0]
    valid = jnp.ones(batch_size, dtype=bool)

    # Check each batch element
    for i in range(batch_size):
        threshold = thresholds[i]
        vals = out_vals[i]
        idxs = out_indexs[i]

        # Check >90% of values are >= threshold
        above_threshold = vals >= threshold
        recall = jnp.mean(above_threshold)
        valid = valid.at[i].set(valid[i] & (recall > 0.9))

        # Check all indices are unique
        unique_idxs = jnp.unique(idxs, size=k, fill_value=-1)
        all_unique = jnp.all((unique_idxs >= 0) & (unique_idxs < shape[1]))
        valid = valid.at[i].set(valid[i] & all_unique)

        # Check indices map to values correctly
        correct_mapping = jnp.all(operand[i, idxs] == vals)
        valid = valid.at[i].set(valid[i] & correct_mapping)

    assert valid.all(), (
        f"approx_max_k large vocab validation failed for dtype {dtype}, k={k}: "
        f"{int(valid.sum())}/{batch_size} rows passed"
    )
