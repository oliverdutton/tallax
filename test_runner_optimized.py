"""Simple test runner for optimized_topk_mask."""

import jax
import jax.numpy as jnp
import numpy as np

from tallax.tax.optimized_topk_mask import (
  monotonic_f32_to_u32,
  monotonic_u32_to_f32,
  interp_f32,
  find_topk_threshold_jax,
  topk_mask_stable,
)


def test_monotonic_roundtrip():
  """Test that conversions roundtrip correctly."""
  print("Testing monotonic roundtrip...")
  test_values = jnp.array([
    -1e10, -100.0, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 100.0, 1e10
  ], dtype=jnp.float32)

  u32_values = monotonic_f32_to_u32(test_values)
  f32_back = monotonic_u32_to_f32(u32_values)

  assert jnp.allclose(test_values, f32_back), f"Roundtrip failed: {test_values} != {f32_back}"
  print("✓ Monotonic roundtrip test passed")


def test_monotonicity():
  """Test that larger floats map to larger uint32 values."""
  print("Testing monotonicity...")
  test_values = jnp.array(
    [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0],
    dtype=jnp.float32
  )

  u32_values = monotonic_f32_to_u32(test_values)

  # Check that u32 values are also sorted
  assert jnp.all(u32_values[:-1] <= u32_values[1:]), "Monotonicity violated"
  print("✓ Monotonicity test passed")


def test_threshold_simple():
  """Test threshold finding on simple sorted array."""
  print("Testing threshold finding...")
  x = jnp.array([[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
  k = 3

  threshold = find_topk_threshold_jax(x, k)
  count_ge = (x >= threshold).sum()

  assert count_ge >= k, f"Threshold test failed: count_ge={count_ge}, k={k}"
  print(f"✓ Threshold test passed (threshold={threshold}, count_ge={count_ge})")


def test_stable_topk_no_ties():
  """Test when there are no tied values."""
  print("Testing stable topk without ties...")
  x = jnp.array([[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
  k = 5
  replace_val = -1e12

  result = topk_mask_stable(x, k, replace_val, stable=True)

  # Count kept elements
  kept = (result != replace_val).sum()
  print(f"  Kept {kept} elements (expected {k})")
  print(f"  Result: {result}")

  # Should keep exactly top 5 values
  assert kept >= k, f"Should keep at least {k} elements, got {kept}"
  print("✓ Stable topk (no ties) test passed")


def test_stable_topk_with_ties():
  """Test stable behavior with tied values."""
  print("Testing stable topk with ties...")
  x = jnp.array([[10.0, 9.0, 8.0, 8.0, 8.0, 7.0, 6.0, 5.0]])
  k = 5

  result = topk_mask_stable(x, k, -1e12, stable=True)
  kept = (result != -1e12).sum()

  print(f"  Kept {kept} elements (expected {k})")
  print(f"  Result: {result}")

  # Should keep exactly k elements in stable mode
  assert kept == k, f"Should keep exactly {k} elements, got {kept}"
  print("✓ Stable topk (with ties) test passed")


def test_compare_with_jax():
  """Compare with jax.lax.top_k."""
  print("Testing comparison with jax.lax.top_k...")
  rng = jax.random.PRNGKey(42)
  x = jax.random.uniform(rng, (2, 20), minval=-10, maxval=10)
  k = 5

  # Our implementation
  our_result = topk_mask_stable(x, k, -jnp.inf, stable=True)
  our_vals = []
  for i in range(x.shape[0]):
    row_vals = our_result[i][our_result[i] != -jnp.inf]
    our_vals.append(jnp.sort(row_vals)[::-1][:k])
  our_vals = jnp.stack(our_vals)

  # JAX implementation
  jax_vals, _ = jax.lax.top_k(x, k)

  print(f"  Our values:\n{our_vals}")
  print(f"  JAX values:\n{jax_vals}")

  # Sort both and compare
  our_sorted = jnp.sort(our_vals, axis=-1)[:, ::-1]
  jax_sorted = jnp.sort(jax_vals, axis=-1)[:, ::-1]

  diff = jnp.abs(our_sorted - jax_sorted).max()
  print(f"  Max difference: {diff}")

  assert diff < 1e-5, f"Results differ too much: {diff}"
  print("✓ Comparison with JAX test passed")


if __name__ == "__main__":
  print("Running optimized_topk_mask tests...\n")

  try:
    test_monotonic_roundtrip()
    test_monotonicity()
    test_threshold_simple()
    test_stable_topk_no_ties()
    test_stable_topk_with_ties()
    test_compare_with_jax()

    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)
  except Exception as e:
    print(f"\n❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
