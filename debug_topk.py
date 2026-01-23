"""Debug script to trace topk_mask execution."""

import jax
import jax.numpy as jnp
import math
from tallax.tax.utils import NUM_LANES

# Create simple test case where all top values are ties
rng = jax.random.PRNGKey(42)
x = jnp.ones((2, 256)) * 1.0  # All same value
x = x.at[:, :120].set(1.0)  # Top 120 are 1.0
x = x.at[:, 120:].set(0.5)  # Rest are 0.5
k = 100

print("Test setup:")
print(f"  Shape: {x.shape}, k: {k}")
print(f"  Values: {jnp.unique(x[0])}")
print(f"  Count of 1.0: {(x[0] == 1.0).sum()}")
print(f"  Count of 0.5: {(x[0] == 0.5).sum()}")

# Step 1: Find threshold
threshold_vals, _ = jax.lax.top_k(x, k)
threshold = threshold_vals[:, k - 1 : k]
print(f"\nThreshold: {threshold}")

# Step 2: Count > threshold
num_gt = (x > threshold).sum(axis=1, keepdims=True)
print(f"Num > threshold: {num_gt}")

# Number of == threshold needed
k_eq_needed = k - num_gt
print(f"Num == threshold needed: {k_eq_needed}")

# All 100 should come from the 1.0s (since threshold is 1.0)
# So we need 100 values == 1.0
print(f"\nExpected: Select 100 values == {threshold[0,0]}")
print(f"Available: {(x[0] == threshold[0,0]).sum()} values")

# Now let's manually compute what should happen:
print("\n" + "="*60)
print("Manual computation:")
print("="*60)

batch_size, vocab_size = x.shape
print(f"vocab_size = {vocab_size}")
print(f"NUM_LANES = {NUM_LANES}")

# First find_boundary_chunk call
chunk_size_1 = int(math.sqrt(vocab_size // NUM_LANES) * NUM_LANES)
print(f"\nFirst chunk_size: {chunk_size_1}")
num_chunks_1 = (vocab_size + chunk_size_1 - 1) // chunk_size_1
print(f"Num chunks: {num_chunks_1}")

# Manually compute cumsums for first stage
print(f"\nFirst stage: finding which {chunk_size_1}-sized chunk contains boundary")
for i in range(num_chunks_1):
    start = i * chunk_size_1
    end = min((i + 1) * chunk_size_1, vocab_size)
    chunk_data = x[0, start:end]
    matches = (chunk_data == threshold[0, 0]).sum()
    print(f"  Chunk {i} ([{start}:{end}]): {matches} matches")

# The boundary should be in chunk 0 since we have 120 matches total
# and need 100, all in first 120 positions

# Run the actual implementation
print("\n" + "="*60)
print("Running actual implementation:")
print("="*60)
from tallax.tax.pallas_topk_mask import topk_mask_pallas
result = topk_mask_pallas(x, k, replace_val=-jnp.inf, stable=True, interpret=True)

for i in range(batch_size):
    count = (result[i] != -jnp.inf).sum()
    print(f"Batch {i}: {count} values (expected {k})")

    if count != k:
        # Find what values we got
        kept_vals = result[i][result[i] != -jnp.inf]
        print(f"  Kept values: {jnp.unique(kept_vals)}")
        print(f"  Counts: {[(v, (kept_vals == v).sum()) for v in jnp.unique(kept_vals)]}")
