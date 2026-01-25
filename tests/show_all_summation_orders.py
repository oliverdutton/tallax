"""Show all possible summation orders for 4 f32 values.

Demonstrates that different summation orders can give different results
in floating-point arithmetic due to rounding.
"""

import jax.numpy as jnp
import itertools


def compute_sum_with_order(values, order_description, computation):
    """Compute sum with specific order and show the result."""
    result = computation()
    return result, order_description


def show_all_summation_orders():
    """Show all possible summation orders for our 4 values."""

    # The 4 values from our example
    v0 = jnp.float32(0.4)
    v1 = jnp.float32(0.3)
    v2 = jnp.float32(0.2)
    v3 = jnp.float32(0.1)

    values = [v0, v1, v2, v3]

    print("="*70)
    print("All Summation Orders for [0.4, 0.3, 0.2, 0.1]")
    print("="*70)
    print(f"\nValues as f32:")
    print(f"  v0 = {v0:.15f}")
    print(f"  v1 = {v1:.15f}")
    print(f"  v2 = {v2:.15f}")
    print(f"  v3 = {v3:.15f}")

    print(f"\n{'='*70}")
    print("Different summation orders:")
    print(f"{'='*70}")

    # All 2^3 = 8 different binary tree structures for summing 4 values
    # Each represents a different associativity grouping

    orders = [
        # Left-associative (sequential left-to-right)
        ("((v0+v1)+v2)+v3", lambda: ((v0 + v1) + v2) + v3),

        # Right-associative (sequential right-to-left)
        ("v0+(v1+(v2+v3))", lambda: v0 + (v1 + (v2 + v3))),

        # Balanced binary trees
        ("(v0+v1)+(v2+v3)", lambda: (v0 + v1) + (v2 + v3)),
        ("(v0+v2)+(v1+v3)", lambda: (v0 + v2) + (v1 + v3)),
        ("(v0+v3)+(v1+v2)", lambda: (v0 + v3) + (v1 + v2)),

        # Other groupings
        ("(v0+(v1+v2))+v3", lambda: (v0 + (v1 + v2)) + v3),
        ("v0+((v1+v2)+v3)", lambda: v0 + ((v1 + v2) + v3)),
        ("((v0+v2)+v1)+v3", lambda: ((v0 + v2) + v1) + v3),
    ]

    results = []
    for desc, computation in orders:
        result = computation()
        results.append((desc, result))

    # Sort by result to show which ones are equal
    results.sort(key=lambda x: x[1])

    print(f"\nResults (sorted by value):\n")
    print(f"{'Order':<25} {'Result':<25} {'Difference from 0.9':<20}")
    print(f"{'-'*25} {'-'*25} {'-'*20}")

    for desc, result in results:
        diff = result - 0.9
        print(f"{desc:<25} {result:.15f}  {diff:+.15e}")

    # Find unique values (convert to float for hashing)
    unique_results = sorted(set(float(r[1]) for r in results))

    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"{'='*70}")
    print(f"Number of different summation orders tested: {len(orders)}")
    print(f"Number of unique results: {len(unique_results)}")
    print(f"\nUnique results:")
    for i, val in enumerate(unique_results, 1):
        count = sum(1 for _, r in results if float(r) == val)
        print(f"  {i}. {val:.15f} (appears {count} times)")

    # Show the range
    if len(unique_results) > 1:
        min_val = min(unique_results)
        max_val = max(unique_results)
        range_val = max_val - min_val

        print(f"\nRange of results: {range_val:.15e}")
        print(f"  Minimum: {min_val:.15f}")
        print(f"  Maximum: {max_val:.15f}")

        print(f"\nMathematical result: 0.9 exactly")
        print(f"All f32 results differ from 0.9 by: {min_val - 0.9:.15e} to {max_val - 0.9:.15e}")


def show_cumulative_sums():
    """Show cumulative sums in different orders."""

    v0 = jnp.float32(0.4)
    v1 = jnp.float32(0.3)
    v2 = jnp.float32(0.2)
    v3 = jnp.float32(0.1)

    print(f"\n{'='*70}")
    print("Cumulative Sums (as done in top-p)")
    print(f"{'='*70}")

    # Different orderings
    orderings = [
        ("Original [0.4, 0.3, 0.2, 0.1]", [v0, v1, v2, v3]),
        ("Sorted desc [0.4, 0.3, 0.2, 0.1]", [v0, v1, v2, v3]),
        ("Reversed [0.1, 0.2, 0.3, 0.4]", [v3, v2, v1, v0]),
        ("Alternating [0.4, 0.2, 0.3, 0.1]", [v0, v2, v1, v3]),
    ]

    for desc, values in orderings:
        print(f"\n{desc}:")
        cumsum = jnp.cumsum(jnp.array(values))
        for i, (val, cs) in enumerate(zip(values, cumsum)):
            marker = " <- exactly 0.9!" if abs(cs - 0.9) < 1e-10 else ""
            marker2 = " <- rounds to > 0.9" if cs > 0.9 else ""
            print(f"  After adding {val:.1f}: cumsum = {cs:.15f}{marker}{marker2}")


def show_first_three_values():
    """Focus on just the first 3 values that sum to 0.9."""

    v0 = jnp.float32(0.4)
    v1 = jnp.float32(0.3)
    v2 = jnp.float32(0.2)

    print(f"\n{'='*70}")
    print("All orders for first 3 values [0.4, 0.3, 0.2]")
    print(f"{'='*70}")

    # All 6 permutations
    import itertools

    print(f"\nAll permutations and their sums:\n")
    print(f"{'Permutation':<25} {'Sum':<25} {'Difference from 0.9':<20}")
    print(f"{'-'*25} {'-'*25} {'-'*20}")

    for perm in itertools.permutations([v0, v1, v2]):
        # Compute sum left-to-right
        result = (perm[0] + perm[1]) + perm[2]
        perm_str = f"({perm[0]:.1f}+{perm[1]:.1f})+{perm[2]:.1f}"
        diff = result - 0.9
        print(f"{perm_str:<25} {result:.15f}  {diff:+.15e}")


def show_sum_of_three_all_orders():
    """Show all 2^2 = 4 different binary tree orderings for summing 3 values."""

    v0 = jnp.float32(0.4)
    v1 = jnp.float32(0.3)
    v2 = jnp.float32(0.2)

    print(f"\n{'='*70}")
    print("All 2^2 = 4 binary tree orderings for summing 3 values to 0.9")
    print(f"{'='*70}")
    print(f"\nValues: v0={v0:.1f}, v1={v1:.1f}, v2={v2:.1f}")
    print(f"Mathematical sum: 0.9 (exact)")

    # All possible binary tree structures for 3 values
    orderings = [
        ("(v0+v1)+v2  [left-assoc]", (v0 + v1) + v2),
        ("v0+(v1+v2)  [right-assoc]", v0 + (v1 + v2)),
        ("(v0+v2)+v1  [reordered]", (v0 + v2) + v1),
        ("(v1+v2)+v0  [reversed]", (v1 + v2) + v0),
    ]

    print(f"\nDifferent binary tree orderings:\n")
    print(f"{'Expression':<30} {'F32 Result':<25} {'Diff from 0.9':<20} {'> 0.9?':<10}")
    print(f"{'-'*30} {'-'*25} {'-'*20} {'-'*10}")

    for desc, result in orderings:
        diff = float(result) - 0.9
        greater = "YES" if result > 0.9 else "NO"
        print(f"{desc:<30} {result:.15f}  {diff:+.15e}  {greater:<10}")

    # Check uniqueness
    unique_vals = set(float(r) for _, r in orderings)

    print(f"\n{'='*70}")
    print(f"Number of unique f32 results: {len(unique_vals)}")

    if len(unique_vals) > 1:
        print(f"Different orderings produce different results!")
        min_val = min(unique_vals)
        max_val = max(unique_vals)
        print(f"Range: {max_val - min_val:.15e}")
    else:
        print(f"All orderings produce the same f32 result: {list(unique_vals)[0]:.15f}")
        print(f"Difference from mathematical 0.9: {list(unique_vals)[0] - 0.9:+.15e}")


if __name__ == "__main__":
    show_sum_of_three_all_orders()  # Most important - show the 0.9 case
    show_all_summation_orders()      # Full 4-value sums
    show_cumulative_sums()
    show_first_three_values()

    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    print("\nDifferent summation orders give different f32 results!")
    print("All results are close to 0.9, but none are exactly 0.9.")
    print("\nThis is why the binary search in top-p masking can give")
    print("different results depending on how values are summed.")
    print(f"{'='*70}\n")
