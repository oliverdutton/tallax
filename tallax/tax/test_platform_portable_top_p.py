"""Tests for platform-portable top-p sampling."""

import jax
import jax.numpy as jnp
import numpy as np

from tallax.tax.high_precision_uint import HighPrecisionUInt
from tallax.tax.platform_portable_top_p import platform_portable_top_p


def test_high_precision_uint_basic():
  """Test basic HighPrecisionUInt operations."""
  print("Testing HighPrecisionUInt basic operations...")

  # Test from_i32_array and to_f32
  x = jnp.array([100, 200, 300], dtype=jnp.int32)
  hp = HighPrecisionUInt.from_i32_array(x)

  # Should have 2 parts (low and high 16 bits)
  assert len(hp.parts) == 2, f"Expected 2 parts, got {len(hp.parts)}"

  # Convert back to f32 and check
  x_f32 = hp.to_f32()
  np.testing.assert_allclose(x_f32, x.astype(jnp.float32), rtol=1e-5)
  print(f"  ✓ from_i32_array and to_f32: {x} -> {x_f32}")

  # Test from_f32
  y_f32 = jnp.array([1000.0, 2000.0, 3000.0], dtype=jnp.float32)
  hp2 = HighPrecisionUInt.from_f32(y_f32, num_parts=4)
  y_roundtrip = hp2.to_f32()
  np.testing.assert_allclose(y_roundtrip, y_f32, rtol=1e-5)
  print(f"  ✓ from_f32 roundtrip: {y_f32} -> {y_roundtrip}")

  print("✓ Basic operations passed\n")


def test_high_precision_uint_sum():
  """Test HighPrecisionUInt summation."""
  print("Testing HighPrecisionUInt sum_dim1...")

  # Create a small test array
  x = jnp.array([[1, 2, 3, 4], [10, 20, 30, 40]], dtype=jnp.int32)
  hp = HighPrecisionUInt.from_i32_array(x)

  # Sum along axis=1
  sum_hp = hp.sum_dim1()
  sum_f32 = sum_hp.to_f32()

  expected = jnp.array([[10], [100]], dtype=jnp.float32)
  np.testing.assert_allclose(sum_f32, expected, rtol=1e-5)
  print(f"  ✓ sum_dim1: {x} -> {sum_f32.squeeze()}")

  # Test with larger values
  x_large = jnp.full((2, 1000), 1000, dtype=jnp.int32)
  hp_large = HighPrecisionUInt.from_i32_array(x_large)
  sum_large_hp = hp_large.sum_dim1()
  sum_large_f32 = sum_large_hp.to_f32()

  expected_large = jnp.array([[1000000], [1000000]], dtype=jnp.float32)
  np.testing.assert_allclose(sum_large_f32, expected_large, rtol=1e-5)
  print(f"  ✓ sum_dim1 large: 1000×1000 = {sum_large_f32.squeeze()}")

  print("✓ Sum operations passed\n")


def test_high_precision_uint_compare():
  """Test HighPrecisionUInt comparison."""
  print("Testing HighPrecisionUInt compare_ge...")

  # Test equal values
  a = HighPrecisionUInt([jnp.array([100, 200]), jnp.array([10, 20])])
  b = HighPrecisionUInt([jnp.array([100, 200]), jnp.array([10, 20])])
  result = a.compare_ge(b)
  assert jnp.all(result), "Equal values should have a >= b"
  print(f"  ✓ compare_ge equal: True")

  # Test greater values
  a = HighPrecisionUInt([jnp.array([100, 200]), jnp.array([20, 20])])
  b = HighPrecisionUInt([jnp.array([100, 200]), jnp.array([10, 20])])
  result = a.compare_ge(b)
  expected = jnp.array([True, True])
  np.testing.assert_array_equal(result, expected)
  print(f"  ✓ compare_ge greater: {result}")

  # Test less than
  a = HighPrecisionUInt([jnp.array([100, 200]), jnp.array([5, 20])])
  b = HighPrecisionUInt([jnp.array([100, 200]), jnp.array([10, 20])])
  result = a.compare_ge(b)
  expected = jnp.array([False, True])
  np.testing.assert_array_equal(result, expected)
  print(f"  ✓ compare_ge less: {result}")

  print("✓ Comparison operations passed\n")


def test_platform_portable_top_p_basic():
  """Test basic top-p functionality."""
  print("Testing platform_portable_top_p basic functionality...")

  # Create simple logits
  logits = jnp.array([
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [5.0, 4.0, 3.0, 2.0, 1.0],
  ], dtype=jnp.float32)

  # Apply top-p with p=0.9
  result = platform_portable_top_p(logits, top_p=0.9)

  # Check that some values are masked
  mask = result != -1e12
  assert jnp.any(mask), "Should have some unmasked values"
  assert jnp.any(~mask), "Should have some masked values"

  print(f"  ✓ Basic masking works")
  print(f"    Logits: {logits[0]}")
  print(f"    Masked: {result[0]}")
  print(f"    Mask: {mask[0]}")

  # Test with top_p=1.0 (should keep all)
  result_all = platform_portable_top_p(logits, top_p=1.0)
  mask_all = result_all != -1e12
  assert jnp.all(mask_all), "top_p=1.0 should keep all values"
  print(f"  ✓ top_p=1.0 keeps all values")

  print("✓ Basic functionality passed\n")


def test_platform_portable_top_p_deterministic():
  """Test that results are deterministic across runs."""
  print("Testing determinism...")

  key = jax.random.PRNGKey(42)
  logits = jax.random.normal(key, (4, 100))

  # Run multiple times
  results = []
  for _ in range(5):
    result = platform_portable_top_p(logits, top_p=0.9)
    results.append(result)

  # Check all results are identical
  for i in range(1, len(results)):
    np.testing.assert_array_equal(results[0], results[i])

  print("  ✓ Results are deterministic across runs")
  print("✓ Determinism test passed\n")


def test_platform_portable_top_p_different_scales():
  """Test with different probability scales."""
  print("Testing different probability scales...")

  # Create logits with very different scales
  logits_small = jnp.array([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=jnp.float32)
  logits_large = jnp.array([[10.0, 20.0, 30.0, 40.0, 50.0]], dtype=jnp.float32)

  result_small = platform_portable_top_p(logits_small, top_p=0.9)
  result_large = platform_portable_top_p(logits_large, top_p=0.9)

  # Both should mask some values
  mask_small = result_small != -1e12
  mask_large = result_large != -1e12

  assert jnp.any(~mask_small), "Small logits should mask some values"
  assert jnp.any(~mask_large), "Large logits should mask some values"

  print(f"  ✓ Works with different scales")
  print(f"    Small logits masked: {(~mask_small).sum()}/5")
  print(f"    Large logits masked: {(~mask_large).sum()}/5")

  print("✓ Scale test passed\n")


def test_platform_portable_top_p_batch():
  """Test with batched inputs and per-sample top_p."""
  print("Testing batched inputs...")

  # Create batch of logits
  key = jax.random.PRNGKey(123)
  logits = jax.random.normal(key, (8, 50))

  # Scalar top_p
  result_scalar = platform_portable_top_p(logits, top_p=0.95)
  assert result_scalar.shape == logits.shape
  print(f"  ✓ Scalar top_p works with batch")

  # Per-sample top_p
  top_p_per_sample = jnp.linspace(0.5, 1.0, 8)
  result_per_sample = platform_portable_top_p(logits, top_p=top_p_per_sample)
  assert result_per_sample.shape == logits.shape

  # Check that different top_p values produce different masks
  masks = result_per_sample != -1e12
  num_kept = masks.sum(axis=1)
  # Lower top_p should keep fewer values
  assert num_kept[0] <= num_kept[-1], "Lower top_p should keep fewer values"
  print(f"  ✓ Per-sample top_p works")
  print(f"    top_p={top_p_per_sample[0]:.2f}: kept {num_kept[0]}/50")
  print(f"    top_p={top_p_per_sample[-1]:.2f}: kept {num_kept[-1]}/50")

  print("✓ Batch test passed\n")


def test_platform_portable_top_p_large_vocab():
  """Test with large vocabulary size."""
  print("Testing large vocabulary...")

  key = jax.random.PRNGKey(456)
  logits = jax.random.normal(key, (2, 32000))  # GPT-like vocab size

  result = platform_portable_top_p(logits, top_p=0.9)
  mask = result != -1e12

  num_kept = mask.sum(axis=1)
  print(f"  ✓ Large vocab (32k) works")
  print(f"    Kept: {num_kept[0]}/{logits.shape[1]} and {num_kept[1]}/{logits.shape[1]}")

  # Check that a reasonable number are kept (not all, not none)
  assert jnp.all(num_kept > 0), "Should keep some values"
  assert jnp.all(num_kept < logits.shape[1]), "Should mask some values"

  print("✓ Large vocab test passed\n")


def run_all_tests():
  """Run all tests."""
  print("="*60)
  print("Running Platform-Portable Top-P Tests")
  print("="*60 + "\n")

  test_high_precision_uint_basic()
  test_high_precision_uint_sum()
  test_high_precision_uint_compare()
  test_platform_portable_top_p_basic()
  test_platform_portable_top_p_deterministic()
  test_platform_portable_top_p_different_scales()
  test_platform_portable_top_p_batch()
  test_platform_portable_top_p_large_vocab()

  print("="*60)
  print("✅ All tests passed!")
  print("="*60)


if __name__ == "__main__":
  run_all_tests()
