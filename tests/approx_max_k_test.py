import pytest
import jax
import jax.numpy as jnp
from tallax import tax
from tallax._src.utils import is_cpu_platform
from tallax._src.test_utils import verify_topk_output
from tallax.divide_and_filter_topk_convergence_theory import (
    compute_expected_recall_at_depth,
    compute_required_depth_for_recall_target,
    compute_cdf_at_depth,
)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32])
@pytest.mark.parametrize("k", [1, 2, 3, 17, 32, 64, 128])
@pytest.mark.parametrize("recall_target", [0.2, 0.5, 0.8, 0.95, 0.99])
@pytest.mark.parametrize("shape", [(128, 8192)])
@pytest.mark.parametrize("use_lax_approx_max_k_algorithm", [False, True])
@pytest.mark.parametrize("seed", [0, 42, 467])
@pytest.mark.skipif(is_cpu_platform(), reason="approx_max_k tests require TPU/GPU")
def test_approx_max_k(dtype, k, recall_target, shape, use_lax_approx_max_k_algorithm, seed):
    """Test approx_max_k with iota reshaped."""
    key = jax.random.key(seed)
    if dtype == jnp.float32:
        operand = jax.random.normal(key, shape, dtype=dtype)
    else:
        operand = jax.random.randint(key, shape, 0, 2**24, dtype=dtype)

    outputs = tax.approx_max_k(operand, k=k, recall_target=recall_target, use_lax_approx_max_k_algorithm=use_lax_approx_max_k_algorithm)
    recall = verify_topk_output(operand, outputs, axis=1, approximate=True).mean()
    # you'd expect checking against recall target to fail 50% of the time, but in practice the approximation bounds are quite strong so this passes 100% of current tests happily
    test_recall_threshold = recall_target
    print(f'recall of {recall:.3f} for target {recall_target:.3f}')
    if k==1:
        assert recall==1., f"For k=1 recall must be perfect, but found {recall:.3f}"
    assert (recall > test_recall_threshold), (
        f"approx_max_k validation failed for dtype {dtype}, k={k}: "
        f"recall={recall:.3f}"
    )


@pytest.mark.parametrize("k", [1, 5, 10, 32, 64])
@pytest.mark.parametrize("num_bins", [128, 256])
def test_expected_recall_at_depth(k, num_bins):
    """Test expected recall computation at different depths."""
    # Test basic properties

    # Recall should increase with depth
    prev_recall = 0.0
    for m in range(1, min(k + 1, 10)):
        recall = compute_expected_recall_at_depth(m, k, num_bins)
        assert recall >= prev_recall, f"Recall should be monotonically increasing, but got {recall} < {prev_recall} at depth {m}"
        assert 0 <= recall <= k, f"Expected recall should be between 0 and k={k}, got {recall}"
        prev_recall = recall

    # At depth 1, we should have some non-zero recall for reasonable num_bins
    recall_depth_1 = compute_expected_recall_at_depth(1, k, num_bins)
    assert recall_depth_1 > 0, "Expected recall at depth 1 should be > 0"

    # At depth k, we should have high recall (close to k)
    recall_depth_k = compute_expected_recall_at_depth(k, k, num_bins)
    assert recall_depth_k > 0.9 * k, f"Expected recall at depth k should be close to k, got {recall_depth_k} vs k={k}"


@pytest.mark.parametrize("k", [5, 10, 32])
@pytest.mark.parametrize("num_bins", [128, 256])
@pytest.mark.parametrize("recall_target", [0.5, 0.8, 0.95, 0.99])
def test_required_depth_for_recall_target(k, num_bins, recall_target):
    """Test computing required depth for a recall target."""
    required_depth = compute_required_depth_for_recall_target(k, num_bins, recall_target)

    # Verify the depth is valid
    assert 1 <= required_depth <= k, f"Depth should be between 1 and k={k}, got {required_depth}"

    # Verify the recall at this depth meets the target
    actual_recall = compute_expected_recall_at_depth(required_depth, k, num_bins)
    expected_recall = recall_target * k
    assert actual_recall >= expected_recall - 1e-6, (
        f"Recall at depth {required_depth} should be >= {expected_recall}, got {actual_recall}"
    )

    # Verify this is the minimum depth (if not at depth 1)
    if required_depth > 1:
        prev_recall = compute_expected_recall_at_depth(required_depth - 1, k, num_bins)
        assert prev_recall < expected_recall, (
            f"Depth {required_depth} should be minimal, but depth {required_depth-1} "
            f"already achieves recall {prev_recall} >= {expected_recall}"
        )


@pytest.mark.parametrize("m", [1, 2, 5])
@pytest.mark.parametrize("j", [1, 5, 10])
@pytest.mark.parametrize("num_bins", [128, 256])
def test_cdf_at_depth(m, j, num_bins):
    """Test CDF computation at different depths."""
    cdf_val = compute_cdf_at_depth(m, j, num_bins)

    # CDF should be a valid probability
    assert 0 <= cdf_val <= 1, f"CDF should be between 0 and 1, got {cdf_val}"

    # CDF should increase with depth
    if m < j:
        cdf_next = compute_cdf_at_depth(m + 1, j, num_bins)
        assert cdf_next >= cdf_val - 1e-10, f"CDF should be monotonically increasing with depth"

    # Edge case: depth 0 should give 0
    assert compute_cdf_at_depth(0, j, num_bins) == 0.0

    # Edge case: j=0 should give 1.0
    assert compute_cdf_at_depth(m, 0, num_bins) == 1.0
