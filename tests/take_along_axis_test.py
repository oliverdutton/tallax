"""
Tests for the new take_along_axis function in the top1 branch.

This test file validates the new take_along_axis (renamed from gather)
and the top1 function.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from tallax import tax
from tallax.tax.gather import pallas_compatible_take_along_axis
from tallax.tax.bitonic_topk import top1
from tallax.utils import is_cpu_platform, NUM_SUBLANES, NUM_LANES


class TestTakeAlongAxis:
    """Test suite for take_along_axis function."""

    @pytest.mark.parametrize("axis", [0, 1])
    def test_take_along_axis_basic(self, axis):
        """Test basic take_along_axis functionality against numpy."""
        # Create simple test data
        values = jnp.arange(20.0).reshape(4, 5)

        if axis == 0:
            # Gather along rows
            indices = jnp.array([[0, 1, 2, 0, 3],
                                [1, 2, 3, 1, 2],
                                [2, 3, 0, 2, 1],
                                [3, 0, 1, 3, 0]])
        else:
            # Gather along columns
            indices = jnp.array([[0, 2, 1, 4, 3],
                                [1, 0, 3, 2, 4],
                                [2, 1, 0, 3, 4],
                                [0, 1, 2, 3, 4]])

        # Get expected result from numpy
        expected = np.take_along_axis(np.array(values), np.array(indices), axis=axis)

        # Get result from our implementation
        result = pallas_compatible_take_along_axis(values, indices, axis=axis)

        # Compare
        np.testing.assert_allclose(
            result, expected,
            err_msg=f"take_along_axis failed for axis={axis}"
        )

    @pytest.mark.parametrize("batch_size,vocab_size,k", [
        (8, 128, 64),
        (16, 256, 128),
        (13, 300, 200),
    ])
    def test_take_along_axis_axis1_compatibility(self, batch_size, vocab_size, k):
        """
        Test that take_along_axis with axis=1 produces same results as old gather.
        This ensures backward compatibility.
        """
        key = jax.random.PRNGKey(0)
        key_vals, key_idxs = jax.random.split(key)

        values = jax.random.normal(key_vals, (batch_size, vocab_size))
        indices = jax.random.randint(key_idxs, (batch_size, k), 0, vocab_size)

        # Expected result using jax.numpy
        expected = jax.vmap(lambda v, i: v[i])(values, indices)

        # Test using interpret mode for CPU compatibility
        result = tax.take_along_axis(values, indices, axis=1, interpret=is_cpu_platform())

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_take_along_axis_axis0(self):
        """Test take_along_axis along axis=0."""
        batch_size = 16
        k = 128

        key = jax.random.PRNGKey(42)
        key_vals, key_idxs = jax.random.split(key)

        values = jax.random.normal(key_vals, (batch_size, k))
        # For each column, pick a row
        indices = jax.random.randint(key_idxs, (batch_size, k), 0, batch_size)

        # Expected result
        expected = jnp.take_along_axis(values, indices, axis=0)

        # Our result
        result = tax.take_along_axis(values, indices, axis=0, interpret=is_cpu_platform())

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_pallas_compatible_small_example(self):
        """
        Small test case to debug the mask logic.
        This will likely fail if Bug #1 in bug_analysis_top1_branch.md is present.
        """
        # Very simple case
        values = jnp.array([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0]])
        indices = jnp.array([[2, 0, 1],
                            [1, 2, 0]])

        # Expected: gather each row using its indices
        # Row 0: [3.0, 1.0, 2.0]
        # Row 1: [5.0, 6.0, 4.0]
        expected = jnp.array([[3.0, 1.0, 2.0],
                             [5.0, 6.0, 4.0]])

        result = pallas_compatible_take_along_axis(values, indices, axis=1)

        # This test may fail due to the masking bug
        np.testing.assert_allclose(
            result, expected,
            err_msg="Simple gather along axis=1 failed - likely due to mask bug"
        )


class TestTop1:
    """Test suite for top1 function."""

    def test_top1_basic(self):
        """Test that top1 returns the maximum along axis 0."""
        # Create data where we know the max
        size = 16  # Must be power of 2 >= NUM_SUBLANES
        width = 256

        key = jax.random.PRNGKey(42)
        values = jax.random.normal(key, (size, width))
        indices = jnp.arange(width)[None, :].repeat(size, axis=0)

        # Get top1
        result_values, result_indices = top1([values, indices], num_keys=1, axis=0)

        # Expected
        expected_values = jnp.max(values, axis=0, keepdims=True)
        expected_indices = jnp.argmax(values, axis=0, keepdims=True)

        # Check values match
        np.testing.assert_allclose(
            result_values, expected_values, rtol=1e-5,
            err_msg="top1 values don't match expected max"
        )

        # Check indices match
        assert jnp.all(result_indices == expected_indices), \
            "top1 indices don't match expected argmax"

    def test_top1_output_shape(self):
        """Test that top1 returns correct output shape."""
        size = 16
        width = 128

        values = jax.random.normal(jax.random.PRNGKey(0), (size, width))
        indices = jnp.arange(width)[None, :].repeat(size, axis=0)

        result_values, result_indices = top1([values, indices], num_keys=1, axis=0)

        # Output should be (1, width)
        assert result_values.shape == (1, width), \
            f"Expected shape (1, {width}), got {result_values.shape}"
        assert result_indices.shape == (1, width), \
            f"Expected shape (1, {width}), got {result_indices.shape}"

    def test_top1_with_gumbel_sampling(self):
        """
        Test top1 in a sampling scenario (as used in top_p_and_sample).
        This simulates the Gumbel-max trick for sampling.
        """
        k = 128
        batch_size = 16

        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)

        # Logits shape (k, batch) - transposed format
        logits = jax.random.normal(key1, (k, batch_size))
        indices = jnp.arange(k)[:, None].repeat(batch_size, axis=1)

        # Add Gumbel noise
        u = jax.random.uniform(key2, logits.shape, minval=1e-10, maxval=1.0)
        gumbel = -jnp.log(-jnp.log(u))
        gumbel_logits = logits + gumbel

        # Get top1
        _, sampled_indices = top1([gumbel_logits, indices], num_keys=1, axis=0)

        # Verify shape
        assert sampled_indices.shape == (1, batch_size), \
            f"Expected shape (1, {batch_size}), got {sampled_indices.shape}"

        # Verify indices are in valid range
        assert jnp.all((sampled_indices >= 0) & (sampled_indices < k)), \
            "Sampled indices out of range"


class TestTopPAndSampleAxes:
    """Test axes and shapes in top_p_and_sample."""

    def test_transpose_logic(self):
        """
        Verify the transpose logic in top_p_and_sample_jax_inner.
        Tests that operations work correctly after transposing from (batch, k) to (k, batch).
        """
        from tallax.tax.fused_sampling import top_p_and_sample_jax_inner

        batch_size = 16
        k = 128
        vocab_size = 1024

        key = jax.random.PRNGKey(42)
        rng_key = jax.random.PRNGKey(123)

        # Create inputs in (batch, k) format
        topk_logits = jax.random.normal(key, (batch_size, k))
        topk_idx = jax.random.randint(key, (batch_size, k), 0, vocab_size)
        top_p = jnp.full((batch_size,), 0.9)
        temperature = jnp.full((batch_size,), 1.0)

        # Run sampling
        try:
            sampled_tokens = top_p_and_sample_jax_inner(
                topk_logits=topk_logits,
                topk_idx=topk_idx,
                rng_key=rng_key,
                top_p=top_p,
                temperature=temperature,
                vocab_size=vocab_size,
                replace_val=-float('inf')
            )

            # Check output shape
            assert sampled_tokens.shape == (batch_size,), \
                f"Expected shape ({batch_size},), got {sampled_tokens.shape}"

            # Check values are valid
            assert jnp.all((sampled_tokens >= 0) & (sampled_tokens < vocab_size)), \
                "Sampled tokens out of valid range"

        except Exception as e:
            pytest.fail(f"top_p_and_sample_jax_inner failed with error: {e}")


if __name__ == "__main__":
    # Run tests directly
    import sys

    print("Testing take_along_axis...")
    test_class = TestTakeAlongAxis()
    try:
        test_class.test_take_along_axis_basic(axis=1)
        print("✓ test_take_along_axis_basic(axis=1) passed")
    except AssertionError as e:
        print(f"✗ test_take_along_axis_basic(axis=1) failed: {e}")
        sys.exit(1)

    try:
        test_class.test_take_along_axis_basic(axis=0)
        print("✓ test_take_along_axis_basic(axis=0) passed")
    except AssertionError as e:
        print(f"✗ test_take_along_axis_basic(axis=0) failed: {e}")
        sys.exit(1)

    try:
        test_class.test_pallas_compatible_small_example()
        print("✓ test_pallas_compatible_small_example passed")
    except AssertionError as e:
        print(f"✗ test_pallas_compatible_small_example failed: {e}")
        print("This likely indicates Bug #1 (mask logic) is present")
        sys.exit(1)

    print("\nTesting top1...")
    test_class2 = TestTop1()
    try:
        test_class2.test_top1_basic()
        print("✓ test_top1_basic passed")
    except AssertionError as e:
        print(f"✗ test_top1_basic failed: {e}")
        sys.exit(1)

    try:
        test_class2.test_top1_output_shape()
        print("✓ test_top1_output_shape passed")
    except AssertionError as e:
        print(f"✗ test_top1_output_shape failed: {e}")
        sys.exit(1)

    print("\nAll tests passed!")
