"""Tests for Pallas topk_mask with parallel chunk-based reduction."""

import jax
import jax.numpy as jnp
import numpy as np

from tallax.tax.pallas_topk_mask import topk_mask_pallas, find_boundary_chunk


def test_find_boundary_chunk_basic():
  """Test find_boundary_chunk with simple case."""
  print("Test: find_boundary_chunk basic")

  # Create logits with known pattern: [5, 5, 5, 5, 3, 3, 3, 1, 1, 1, 1, 1]
  logits = jnp.array([[5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
  target = jnp.array([[5.0]])  # Looking for matches of value 5
  k = jnp.array([[2]], dtype=jnp.int32)  # Want 2nd occurrence
  chunk_size = 4

  chunk_idx, cumsum_before = find_boundary_chunk(logits, target, k, chunk_size)

  # First chunk (0-3) has 4 matches, so 2nd match is in first chunk (idx 0)
  print(f"  chunk_idx: {chunk_idx}, cumsum_before: {cumsum_before}")
  assert chunk_idx[0, 0] == 0, f"Expected chunk 0, got {chunk_idx[0, 0]}"
  assert cumsum_before[0, 0] == 0, f"Expected cumsum_before 0, got {cumsum_before[0, 0]}"
  print("  PASS\n")


def test_find_boundary_chunk_across_chunks():
  """Test find_boundary_chunk when boundary crosses chunks."""
  print("Test: find_boundary_chunk across chunks")

  # Create logits: [7, 7, 5, 5, 5, 5, 5, 5, 3, 3]
  logits = jnp.array([[7.0, 7.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 3.0, 3.0]])
  target = jnp.array([[5.0]])  # Looking for matches of value 5
  k = jnp.array([[3]], dtype=jnp.int32)  # Want 3rd occurrence
  chunk_size = 4

  chunk_idx, cumsum_before = find_boundary_chunk(logits, target, k, chunk_size)

  # Chunk 0 (0-3): 2 matches at positions 2, 3
  # Chunk 1 (4-7): 4 matches at positions 4, 5, 6, 7
  # 3rd match is in chunk 1
  print(f"  chunk_idx: {chunk_idx}, cumsum_before: {cumsum_before}")
  assert chunk_idx[0, 0] == 1, f"Expected chunk 1, got {chunk_idx[0, 0]}"
  assert cumsum_before[0, 0] == 2, f"Expected cumsum_before 2, got {cumsum_before[0, 0]}"
  print("  PASS\n")


def test_find_boundary_chunk_batched():
  """Test find_boundary_chunk with batched inputs."""
  print("Test: find_boundary_chunk batched")

  # Two batches with different patterns
  logits = jnp.array([
    [5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0, 1.0],  # Batch 0
    [7.0, 7.0, 5.0, 5.0, 5.0, 5.0, 3.0, 3.0],  # Batch 1
  ])
  target = jnp.array([[5.0], [5.0]])
  k = jnp.array([[2], [3]], dtype=jnp.int32)
  chunk_size = 4

  chunk_idx, cumsum_before = find_boundary_chunk(logits, target, k, chunk_size)

  print(f"  Batch 0 - chunk_idx: {chunk_idx[0, 0]}, cumsum_before: {cumsum_before[0, 0]}")
  print(f"  Batch 1 - chunk_idx: {chunk_idx[1, 0]}, cumsum_before: {cumsum_before[1, 0]}")

  # Batch 0: 4 matches in chunk 0, so 2nd match is in chunk 0
  assert chunk_idx[0, 0] == 0, f"Batch 0: expected chunk 0, got {chunk_idx[0, 0]}"

  # Batch 1: 2 matches in chunk 0, 4 matches in chunk 1, so 3rd match is in chunk 1
  assert chunk_idx[1, 0] == 1, f"Batch 1: expected chunk 1, got {chunk_idx[1, 0]}"
  assert cumsum_before[1, 0] == 2, f"Batch 1: expected cumsum_before 2, got {cumsum_before[1, 0]}"

  print("  PASS\n")


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
    # Test find_boundary_chunk
    test_find_boundary_chunk_basic()
    test_find_boundary_chunk_across_chunks()
    test_find_boundary_chunk_batched()

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
