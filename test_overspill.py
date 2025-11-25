"""
Test to verify vocab_size overspill handling in blockwise_topk.

This test verifies that when vocab_size doesn't divide num_blocks evenly,
the remaining elements are properly handled.
"""

import jax
import jax.numpy as jnp
from tallax import tax
from tallax.utils import is_cpu_platform

# Test case where vocab_size doesn't divide num_blocks
k = 64
num_queries = 8
vocab_size = 135  # Doesn't divide 128 evenly (135 = 1*128 + 7)

logit_key = jax.random.key(0)
logits = jax.random.normal(
    logit_key, (num_queries, vocab_size), dtype=jnp.float32
).astype(jnp.bfloat16)

topk_xla = jax.jit(jax.lax.top_k, static_argnames=("k",))

def test_overspill():
  interpret = is_cpu_platform()
  print(f'Testing topk with vocab_size={vocab_size} (not divisible by 128)')
  print(f'logits shape: {logits.shape}, dtype: {logits.dtype}, k={k}')

  # Get reference result from XLA
  xla_values, xla_indices = topk_xla(logits, k=k)
  print(f"\nXLA result shapes: values={xla_values.shape}, indices={xla_indices.shape}")

  # Get result from Pallas implementation
  pallas_values, pallas_indices = tax.top_k(logits, k=k, block_size=8, interpret=interpret)
  print(f"Pallas result shapes: values={pallas_values.shape}, indices={pallas_indices.shape}")

  # Compare results
  values_match = (xla_values == pallas_values).mean()
  indices_match = (xla_indices == pallas_indices).mean()

  print(f"\nValues match rate: {values_match:.4f}")
  print(f"Indices match rate: {indices_match:.4f}")

  if values_match > 0.99 and indices_match > 0.99:
    print("\n✓ Test PASSED: Overspill handling works correctly!")
  else:
    print("\n✗ Test FAILED: Results don't match!")
    print("\nFirst query XLA values:", xla_values[0])
    print("First query Pallas values:", pallas_values[0])
    print("\nFirst query XLA indices:", xla_indices[0])
    print("First query Pallas indices:", pallas_indices[0])

if __name__ == "__main__":
  test_overspill()
