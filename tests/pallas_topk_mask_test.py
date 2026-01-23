"""Tests for Pallas topk_mask with parallel chunk-based reduction."""

import jax
import jax.numpy as jnp
import numpy as np

from tallax.tax.pallas_topk_mask import topk_mask_pallas


def test_simple_topk():
  """Test basic topk functionality."""
  print("Test 1: Simple topk")

  x = jnp.array([[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0]])
  k = 5

  result = topk_mask_pallas(x, k, replace_val=-jnp.inf, stable=True, interpret=True)

  # Should keep top 5 values
  kept = (result != -jnp.inf).sum()
  print(f"  Kept {kept} values (expected {k})")
  assert kept == k, f"Expected {k} values, got {kept}"
  print("  PASS\n")


def test_with_ties():
  """Test stable behavior with ties."""
  print("Test 2: Topk with ties")

  x = jnp.array([[10.0, 8.0, 8.0, 8.0, 5.0, 3.0, 1.0, 0.0]])
  k = 4

  result = topk_mask_pallas(x, k, replace_val=-jnp.inf, stable=True, interpret=True)

  # Should keep exactly k values
  kept = (result != -jnp.inf).sum()
  print(f"  Kept {kept} values (expected {k})")
  assert kept == k, f"Expected {k} values, got {kept}"
  print("  PASS\n")


def test_batched():
  """Test batched operation."""
  print("Test 3: Batched topk")

  rng = jax.random.PRNGKey(42)
  x = jax.random.normal(rng, (4, 256))
  k = 20

  result = topk_mask_pallas(x, k, replace_val=-jnp.inf, stable=True, interpret=True)

  # Check each batch
  for i in range(4):
    kept = (result[i] != -jnp.inf).sum()
    assert kept == k, f"Batch {i}: expected {k} values, got {kept}"

  print(f"  All {x.shape[0]} batches have exactly {k} elements")
  print("  PASS\n")


def test_comparison_with_jax():
  """Compare with jax.lax.top_k."""
  print("Test 4: Comparison with JAX top_k")

  rng = jax.random.PRNGKey(42)
  x = jax.random.uniform(rng, (2, 100))
  k = 10

  # Our implementation
  our_result = topk_mask_pallas(x, k, replace_val=-jnp.inf, stable=True, interpret=True)
  our_vals = []
  for i in range(2):
    vals = our_result[i][our_result[i] != -jnp.inf]
    our_vals.append(jnp.sort(vals)[::-1][:k])
  our_vals = jnp.stack(our_vals)

  # JAX implementation
  jax_vals, _ = jax.lax.top_k(x, k)

  # Sort and compare
  our_sorted = jnp.sort(our_vals, axis=-1)[:, ::-1]
  jax_sorted = jnp.sort(jax_vals, axis=-1)[:, ::-1]

  diff = jnp.abs(our_sorted - jax_sorted).max()
  print(f"  Max difference from JAX: {diff}")
  assert diff < 1e-5, f"Results differ too much: {diff}"
  print("  PASS\n")


def test_large_vocabulary():
  """Test with large vocabulary size."""
  print("Test 5: Large vocabulary (2048)")

  rng = jax.random.PRNGKey(42)
  x = jax.random.normal(rng, (2, 2048))
  k = 64

  result = topk_mask_pallas(x, k, replace_val=-jnp.inf, stable=True, interpret=True)

  for i in range(2):
    kept = (result[i] != -jnp.inf).sum()
    assert kept == k, f"Batch {i}: expected {k}, got {kept}"

  print(f"  Successfully processed vocab_size={x.shape[1]}")
  print("  PASS\n")


def run_all_tests():
  """Run all tests."""
  print("="*60)
  print("Pallas TopK Mask Tests")
  print("="*60 + "\n")

  try:
    # Test topk_mask_pallas
    test_simple_topk()
    test_with_ties()
    test_batched()
    test_comparison_with_jax()
    test_large_vocabulary()

    print("="*60)
    print("All Pallas topk_mask tests passed!")
    print("="*60)
  except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    return False

  return True


if __name__ == "__main__":
  success = run_all_tests()
  exit(0 if success else 1)
