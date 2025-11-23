
import pytest
import jax
import jax.numpy as jnp
from tallax import tax
from tallax.utils import is_cpu_platform

def test_top_k_precise():
    interpret = is_cpu_platform()
    # Adjust parameters based on platform
    if interpret: # CPU
        vocab_size = 256
        k = 4
        block_size = 1 # Use 1 for small test to avoid padding overhead/confusion
    else: # TPU
        vocab_size = 201088
        k = 64
        block_size = 8 # Default

    num_queries = 8

    key = jax.random.key(42)
    key, subkey = jax.random.split(key)

    # Random values
    random_vals = jax.random.normal(subkey, (num_queries, vocab_size))

    # Create base array with values <= 0
    base_vals = jnp.minimum(random_vals, 0.0)

    # Select k random indices per query
    key, subkey = jax.random.split(key)
    target_indices = jax.random.choice(subkey, vocab_size, shape=(num_queries, k), replace=False)

    # Create values >= 1
    high_vals = jnp.maximum(random_vals, 1.0)

    # Construct input logits: min(random_vals, 0).at[k random ints].set(max(random_vals, 1))
    def update_row(row_vals, row_indices, row_high):
        return row_vals.at[row_indices].set(row_high[row_indices])

    logits = jax.vmap(update_row)(base_vals, target_indices, high_vals)

    # Expected top k are the values we set
    top_vals, top_idxs = tax.top_k(logits, k=k, block_size=block_size, interpret=interpret)

    # Verification
    target_indices_sorted = jnp.sort(target_indices, axis=1)
    top_idxs_sorted = jnp.sort(top_idxs, axis=1)

    assert (target_indices_sorted == top_idxs_sorted).all(), "Indices do not match"

    target_vals = jax.vmap(lambda l, i: l[i])(logits, target_indices)
    target_vals_sorted = jnp.sort(target_vals, axis=1)[:, ::-1]
    top_vals_sorted = jnp.sort(top_vals, axis=1)[:, ::-1]

    assert jnp.allclose(target_vals_sorted, top_vals_sorted), "Values do not match"


def test_top_k_nans():
    interpret = is_cpu_platform()

    logits_nan = jnp.array([
        [1.0, 2.0, jnp.nan, 3.0, 0.0, -1.0],
        [jnp.nan, 5.0, 4.0, 2.0, 1.0, 0.0]
    ], dtype=jnp.float32)

    # Use block_size=2 to match num_rows=2 (divisibility check in top_dynamic_k)
    block_size = 2

    k = 3

    # tax.top_k should handle padding of vocab dimension automatically now
    top_vals_tax, top_idxs_tax = tax.top_k(logits_nan, k=k, block_size=block_size, interpret=interpret)
    top_vals_lax, top_idxs_lax = jax.lax.top_k(logits_nan, k=k)

    # Verify tax.top_k does NOT have NaNs
    assert not jnp.isnan(top_vals_tax).any(), "tax.top_k returned NaNs"

    # Verify lax.top_k HAS NaNs (as per doc comparison)
    assert jnp.isnan(top_vals_lax).any(), "lax.top_k did not return NaNs (unexpected for this test case)"

    # Verify values are correct (ignoring NaNs)
    # Row 0: [1, 2, nan, 3, 0, -1]. Top 3 (ignoring nan): 3, 2, 1.
    expected_vals_0 = jnp.array([3.0, 2.0, 1.0])
    # Row 1: [nan, 5, 4, 2, 1, 0]. Top 3: 5, 4, 2.
    expected_vals_1 = jnp.array([5.0, 4.0, 2.0])

    assert jnp.allclose(top_vals_tax[0], expected_vals_0)
    assert jnp.allclose(top_vals_tax[1], expected_vals_1)

if __name__ == "__main__":
    # Allow running as script
    test_top_k_precise()
    test_top_k_nans()
    print("All tests passed!")
