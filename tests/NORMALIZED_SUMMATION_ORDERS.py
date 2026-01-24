"""Show all 2^3 = 8 summation orderings with NORMALIZED values.

Normalizes [0.4, 0.3, 0.2, 0.1] by their sum first, then shows different orderings.
"""

import jax.numpy as jnp


def show_normalized_summation_orders():
    """Normalize values first, then show all summation orderings."""

    # Start with unnormalized values
    unnorm = jnp.array([0.4, 0.3, 0.2, 0.1], dtype=jnp.float32)

    print("="*80)
    print("NORMALIZED SUMMATION: All 2^3 = 8 orderings")
    print("="*80)

    # Compute sum for normalization
    total = unnorm.sum()

    print(f"\nUnnormalized values: {unnorm}")
    print(f"Sum of unnormalized: {float(total):.15f}")

    # Normalize
    v_norm = unnorm / total

    print(f"\nNormalized values (divided by sum):")
    for i, val in enumerate(v_norm):
        print(f"  v{i} = {float(val):.20f}")

    print(f"\nSum of normalized values: {float(v_norm.sum()):.15f}")

    v = [v_norm[i] for i in range(4)]

    # All 8 binary tree structures
    groupings = [
        ("1. (((v0+v1)+v2)+v3)", lambda: (((v[0]+v[1])+v[2])+v[3])),
        ("2. (v0+(v1+(v2+v3)))", lambda: (v[0]+(v[1]+(v[2]+v[3])))),
        ("3. ((v0+v1)+(v2+v3))", lambda: ((v[0]+v[1])+(v[2]+v[3]))),
        ("4. ((v0+(v1+v2))+v3)", lambda: ((v[0]+(v[1]+v[2]))+v[3])),
        ("5. (v0+((v1+v2)+v3))", lambda: (v[0]+((v[1]+v[2])+v[3]))),
        ("6. ((v0+v2)+(v1+v3))", lambda: ((v[0]+v[2])+(v[1]+v[3]))),
        ("7. (((v0+v2)+v1)+v3)", lambda: (((v[0]+v[2])+v[1])+v[3])),
        ("8. (v0+(v2+(v1+v3)))", lambda: (v[0]+(v[2]+(v[1]+v[3])))),
    ]

    print(f"\n{'='*80}")
    print(f"All 8 summation orderings:")
    print(f"{'='*80}\n")

    print(f"{'#':<3} {'Expression':<25} {'F32 Result':<25} {'Diff from 1.0':<20}")
    print(f"{'-'*3} {'-'*25} {'-'*25} {'-'*20}")

    results = []
    for desc, computation in groupings:
        result = float(computation())
        diff = result - 1.0
        results.append(result)
        print(f"{desc:<28} {result:.15f}  {diff:+.15e}")

    # Check uniqueness
    unique = set(results)

    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"{'='*80}")
    print(f"Number of unique results: {len(unique)}")

    if len(unique) > 1:
        print(f"\nDifferent orderings give different results!")
        for val in sorted(unique):
            count = results.count(val)
            print(f"  {val:.20f} (appears {count} times)")

        vals = sorted(unique)
        print(f"\nRange: {vals[-1] - vals[0]:.15e}")
    else:
        print(f"All orderings give the same result: {list(unique)[0]:.15f}")


def show_normalized_first_three():
    """Show first 3 normalized values summing to 0.9."""

    # Unnormalized
    unnorm = jnp.array([0.4, 0.3, 0.2, 0.1], dtype=jnp.float32)
    total = unnorm.sum()

    # Normalize
    v_norm = unnorm / total

    # Take first 3
    v = [v_norm[i] for i in range(3)]

    print(f"\n{'='*80}")
    print("CRITICAL: First 3 normalized values")
    print(f"{'='*80}")

    print(f"\nFirst 3 normalized values:")
    for i, val in enumerate(v):
        print(f"  v{i} = {float(val):.20f}")

    # Show what they should sum to
    expected_sum = 0.9
    print(f"\nMathematical sum: {expected_sum} (what we expect)")

    # All orderings for 3 values
    groupings = [
        ("(v0+v1)+v2", (v[0]+v[1])+v[2]),
        ("v0+(v1+v2)", v[0]+(v[1]+v[2])),
        ("(v0+v2)+v1", (v[0]+v[2])+v[1]),
        ("(v1+v2)+v0", (v[1]+v[2])+v[0]),
        ("v1+(v0+v2)", v[1]+(v[0]+v[2])),
        ("v2+(v0+v1)", v[2]+(v[0]+v[1])),
        ("(v1+v0)+v2", (v[1]+v[0])+v[2]),
        ("(v2+v1)+v0", (v[2]+v[1])+v[0]),
    ]

    print(f"\nAll summation orderings for first 3 normalized values:")
    print(f"\n{'Expression':<20} {'F32 Result':<25} {'vs 0.9':<20} {'Comparison':<15}")
    print(f"{'-'*20} {'-'*25} {'-'*20} {'-'*15}")

    results_map = {}
    for desc, result in groupings:
        result_val = float(result)
        diff = result_val - expected_sum
        comp = "> 0.9" if result_val > expected_sum else ("= 0.9" if abs(result_val - expected_sum) < 1e-10 else "< 0.9")
        print(f"{desc:<20} {result_val:.15f}  {diff:+.15e}  {comp:<15}")

        if result_val not in results_map:
            results_map[result_val] = []
        results_map[result_val].append(desc)

    # Show unique values
    unique = sorted(results_map.keys())

    print(f"\n{'='*80}")
    print(f"KEY FINDING:")
    print(f"{'='*80}")
    print(f"Number of unique f32 results: {len(unique)}")

    if len(unique) > 1:
        print(f"\nUnique results:")
        for i, val in enumerate(unique, 1):
            count = len(results_map[val])
            comp = "> 0.9" if val > expected_sum else ("< 0.9" if val < expected_sum else "= 0.9")
            print(f"  {i}. {val:.20f} ({comp}, appears {count} times)")
            print(f"      Difference from 0.9: {val - expected_sum:+.15e}")

        print(f"\nRange: {unique[-1] - unique[0]:.15e}")

        print(f"\n*** DIFFERENT SUMMATION ORDERS GIVE DIFFERENT RESULTS! ***")
        print(f"\nThis is why f32 and i32 top-p implementations can differ!")


if __name__ == "__main__":
    show_normalized_summation_orders()
    show_normalized_first_three()

    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    print(f"\nEven after normalization (dividing by sum), floating-point")
    print(f"rounding means different summation orders give different results.")
    print(f"\nThis is fundamental to how f32 works and why high-precision")
    print(f"i32 arithmetic is needed for numerical correctness!")
    print(f"{'='*80}\n")
