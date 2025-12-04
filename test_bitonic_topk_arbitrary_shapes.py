#!/usr/bin/env python3
"""Test bitonic_topk with arbitrary shapes."""

import jax
import jax.numpy as jnp
from tallax.tax.bitonic_topk import bitonic_topk, _compute_padded_shape
from tallax.utils import NUM_LANES


def test_padding_computation():
    """Test that _compute_padded_shape produces correct padding."""
    print("Testing padding computation...")

    # Test case 1: (8, 256) -> (64, 256)
    padded_dim0, padded_dim1 = _compute_padded_shape(8, 256)
    assert padded_dim0 == 64, f"Expected padded_dim0=64, got {padded_dim0}"
    assert padded_dim1 == 256, f"Expected padded_dim1=256, got {padded_dim1}"
    assert (padded_dim0 * padded_dim1) % (NUM_LANES**2) == 0, "Product must be multiple of NUM_LANES**2"
    assert padded_dim1 % NUM_LANES == 0, "dim1 must be multiple of NUM_LANES"
    print(f"  (8, 256) -> ({padded_dim0}, {padded_dim1}) ✓")

    # Test case 2: (8, 8320) -> (8, 10240)
    padded_dim0, padded_dim1 = _compute_padded_shape(8, 8320)
    assert padded_dim0 == 8, f"Expected padded_dim0=8, got {padded_dim0}"
    assert padded_dim1 == 10240, f"Expected padded_dim1=10240, got {padded_dim1}"
    assert (padded_dim0 * padded_dim1) % (NUM_LANES**2) == 0, "Product must be multiple of NUM_LANES**2"
    assert padded_dim1 % NUM_LANES == 0, "dim1 must be multiple of NUM_LANES"
    print(f"  (8, 8320) -> ({padded_dim0}, {padded_dim1}) ✓")

    # Test case 3: Already aligned (8, 128)
    padded_dim0, padded_dim1 = _compute_padded_shape(8, 128)
    assert padded_dim0 == 128, f"Expected padded_dim0=128, got {padded_dim0}"
    assert padded_dim1 == 128, f"Expected padded_dim1=128, got {padded_dim1}"
    print(f"  (8, 128) -> ({padded_dim0}, {padded_dim1}) ✓")

    print("Padding computation tests passed!\n")


def test_bitonic_topk_small_shape():
    """Test bitonic_topk with shape (8, 256)."""
    print("Testing bitonic_topk with shape (8, 256)...")

    # Create test input with known values
    key = jax.random.PRNGKey(42)
    logits = jax.random.normal(key, (8, 256))

    # Run bitonic_topk
    result = bitonic_topk(logits, interpret=True)
    values = result[0]

    # Verify output shape
    assert values.shape == (8, NUM_LANES), f"Expected shape (8, {NUM_LANES}), got {values.shape}"

    # Verify values are sorted descending per row
    for i in range(8):
        row_sorted = jnp.sort(values[i], descending=True)
        assert jnp.allclose(values[i], row_sorted), f"Row {i} not sorted descending"

    # Verify top-k values are correct
    for i in range(8):
        expected_topk = jnp.sort(logits[i], descending=True)[:NUM_LANES]
        assert jnp.allclose(values[i], expected_topk), f"Row {i} top-k values incorrect"

    print(f"  Output shape: {values.shape} ✓")
    print(f"  Values correctly sorted and selected ✓")
    print("Small shape test passed!\n")


def test_bitonic_topk_large_shape():
    """Test bitonic_topk with shape (8, 8320)."""
    print("Testing bitonic_topk with shape (8, 8320)...")

    # Create test input
    key = jax.random.PRNGKey(123)
    logits = jax.random.normal(key, (8, 8320))

    # Run bitonic_topk
    result = bitonic_topk(logits, interpret=True)
    values = result[0]

    # Verify output shape
    assert values.shape == (8, NUM_LANES), f"Expected shape (8, {NUM_LANES}), got {values.shape}"

    # Verify values are sorted descending per row
    for i in range(8):
        row_sorted = jnp.sort(values[i], descending=True)
        assert jnp.allclose(values[i], row_sorted), f"Row {i} not sorted descending"

    # Verify top-k values are correct
    for i in range(8):
        expected_topk = jnp.sort(logits[i], descending=True)[:NUM_LANES]
        assert jnp.allclose(values[i], expected_topk), f"Row {i} top-k values incorrect"

    print(f"  Output shape: {values.shape} ✓")
    print(f"  Values correctly sorted and selected ✓")
    print("Large shape test passed!\n")


def test_bitonic_topk_edge_cases():
    """Test various edge cases."""
    print("Testing edge cases...")

    test_shapes = [
        (1, 128),      # Single token, minimal dim1
        (8, 129),      # Non-aligned dim1
        (16, 1000),    # Non-power-of-2 dims
        (32, 10000),   # Larger input
    ]

    for shape in test_shapes:
        print(f"  Testing shape {shape}...")
        key = jax.random.PRNGKey(hash(shape) % 2**32)
        logits = jax.random.normal(key, shape)

        # Run bitonic_topk
        result = bitonic_topk(logits, interpret=True)
        values = result[0]

        # Verify output shape
        assert values.shape == (shape[0], NUM_LANES), f"Incorrect output shape for {shape}"

        # Verify correctness for first row
        expected_topk = jnp.sort(logits[0], descending=True)[:NUM_LANES]
        assert jnp.allclose(values[0], expected_topk), f"Incorrect values for shape {shape}"
        print(f"    ✓")

    print("Edge case tests passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Bitonic Top-K with Arbitrary Shapes")
    print("=" * 60)
    print()

    test_padding_computation()
    test_bitonic_topk_small_shape()
    test_bitonic_topk_large_shape()
    test_bitonic_topk_edge_cases()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
