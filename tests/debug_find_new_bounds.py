"""Debug find_new_bounds to see what's happening."""

import jax.numpy as jnp


def debug_find_new_bounds():
    """Debug version of find_new_bounds_unrolled."""

    pivots = [jnp.array([[1.0]]), jnp.array([[2.0]]), jnp.array([[3.0]])]
    predicates = [jnp.array([[False]]), jnp.array([[False]]), jnp.array([[True]])]

    num_pivots = len(pivots)

    print("Input:")
    print(f"  num_pivots = {num_pivots}")
    print(f"  pivots shapes: {[p.shape for p in pivots]}")
    print(f"  pivots values: {[float(p[0,0]) for p in pivots]}")
    print(f"  predicates shapes: {[p.shape for p in predicates]}")
    print(f"  predicates values: {[bool(p[0,0]) for p in predicates]}")
    print()

    # Stack arrays
    pivot_array = jnp.stack(pivots)
    pred_array = jnp.stack(predicates)

    print("After stacking:")
    print(f"  pivot_array.shape = {pivot_array.shape}")
    print(f"  pivot_array = {pivot_array}")
    print(f"  pred_array.shape = {pred_array.shape}")
    print(f"  pred_array = {pred_array}")
    print()

    any_true = jnp.any(pred_array)
    any_false = jnp.any(~pred_array)

    print(f"  any_true = {bool(any_true)}")
    print(f"  any_false = {bool(any_false)}")
    print()

    # Find indices
    false_indices = jnp.where(~pred_array, jnp.arange(num_pivots), -1)
    last_false_idx = jnp.max(false_indices)

    true_indices = jnp.where(pred_array, jnp.arange(num_pivots), num_pivots)
    first_true_idx = jnp.min(true_indices)

    print(f"  ~pred_array = {~pred_array}")
    print(f"  jnp.arange(num_pivots) = {jnp.arange(num_pivots)}")
    print(f"  false_indices = {false_indices}")
    print(f"  last_false_idx = {int(last_false_idx)}")
    print()

    print(f"  true_indices = {true_indices}")
    print(f"  first_true_idx = {int(first_true_idx)}")
    print()

    # Safe indexing
    last_false_idx_safe = jnp.clip(last_false_idx, 0, num_pivots - 1)
    first_true_idx_safe = jnp.clip(first_true_idx, 0, num_pivots - 1)

    print(f"  last_false_idx_safe = {int(last_false_idx_safe)}")
    print(f"  first_true_idx_safe = {int(first_true_idx_safe)}")
    print()

    # Index into pivot_array
    print(f"  pivot_array[last_false_idx_safe] = {pivot_array[last_false_idx_safe]}")
    print(f"  pivot_array[first_true_idx_safe] = {pivot_array[first_true_idx_safe]}")
    print()

    new_l_value = jnp.where(any_false, pivot_array[last_false_idx_safe], pivots[0])
    new_r_value = jnp.where(any_true, pivot_array[first_true_idx_safe], pivots[-1])

    print(f"  new_l_value = {new_l_value}")
    print(f"  new_r_value = {new_r_value}")
    print()

    print(f"Expected: new_l=2.0, new_r=3.0")
    print(f"Got: new_l={float(new_l_value[0,0])}, new_r={float(new_r_value[0,0])}")


if __name__ == "__main__":
    debug_find_new_bounds()
