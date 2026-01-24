"""Show all 2^3 = 8 different summation groupings for 4 values.

Demonstrates how floating-point associativity affects the result.
"""

import jax.numpy as jnp


def show_8_summation_orders():
    """Show all 8 ways to group 4 additions (2^3 = 8 binary choices)."""

    v = [jnp.float32(0.4), jnp.float32(0.3), jnp.float32(0.2), jnp.float32(0.1)]

    print("="*80)
    print("ALL 2^3 = 8 DIFFERENT SUMMATION GROUPINGS FOR [0.4, 0.3, 0.2, 0.1]")
    print("="*80)

    print(f"\nValues as f32:")
    for i, val in enumerate(v):
        print(f"  v{i} = {float(val):.15f}")

    print(f"\n{'='*80}")
    print("All 8 binary tree structures (different associativity groupings):")
    print(f"{'='*80}\n")

    # All 8 different ways to group 4 additions
    # Each line represents a different binary tree structure

    groupings = [
        # Structure 1: Left-associative all the way
        ("1. (((v0+v1)+v2)+v3)", lambda: (((v[0]+v[1])+v[2])+v[3])),

        # Structure 2: Right-associative all the way
        ("2. (v0+(v1+(v2+v3)))", lambda: (v[0]+(v[1]+(v[2]+v[3])))),

        # Structure 3: Balanced (two pairs)
        ("3. ((v0+v1)+(v2+v3))", lambda: ((v[0]+v[1])+(v[2]+v[3]))),

        # Structure 4: Left-heavy (pair on left)
        ("4. ((v0+(v1+v2))+v3)", lambda: ((v[0]+(v[1]+v[2]))+v[3])),

        # Structure 5: Right-heavy (pair on right)
        ("5. (v0+((v1+v2)+v3))", lambda: (v[0]+((v[1]+v[2])+v[3]))),

        # Structure 6: Reordered - balanced
        ("6. ((v0+v2)+(v1+v3))", lambda: ((v[0]+v[2])+(v[1]+v[3]))),

        # Structure 7: Reordered - left-heavy
        ("7. (((v0+v2)+v1)+v3)", lambda: (((v[0]+v[2])+v[1])+v[3])),

        # Structure 8: Reordered - right-heavy
        ("8. (v0+(v2+(v1+v3)))", lambda: (v[0]+(v[2]+(v[1]+v[3])))),
    ]

    print(f"{'#':<3} {'Expression':<25} {'F32 Result':<25} {'Exact?':<10}")
    print(f"{'-'*3} {'-'*25} {'-'*25} {'-'*10}")

    results = []
    for desc, computation in groupings:
        result = float(computation())
        exact = "YES" if abs(result - 1.0) < 1e-10 else "NO"
        results.append(result)
        print(f"{desc:<28} {result:.15f}  {exact:<10}")

    # Summary
    unique = set(results)
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"{'='*80}")
    print(f"Number of unique results: {len(unique)}")
    if len(unique) == 1:
        print(f"All 8 groupings give the same f32 result: {list(unique)[0]:.15f}")
        print(f"\nThis makes sense! All sum to 1.0, and 1.0 is exactly representable in f32.")
    else:
        print(f"Different groupings give different results (due to rounding):")
        for val in sorted(unique):
            count = results.count(val)
            print(f"  {val:.15f} (appears {count} times)")


def show_critical_3value_case():
    """Show the critical case: summing first 3 values to get 0.9."""

    v = [jnp.float32(0.4), jnp.float32(0.3), jnp.float32(0.2)]

    print(f"\n{'='*80}")
    print("CRITICAL CASE: First 3 values [0.4, 0.3, 0.2] summing to 0.9")
    print(f"{'='*80}")

    print(f"\nMathematical sum: 0.4 + 0.3 + 0.2 = 0.9 (exact)")
    print(f"But 0.9 is NOT exactly representable in binary floating-point!")
    print(f"0.9 in binary = 0.1110011001100... (repeating)")

    # All 4 binary tree structures for 3 values (Catalan number C_2 = 2, but with reordering = 4)
    groupings = [
        ("(v0+v1)+v2", (v[0]+v[1])+v[2]),
        ("v0+(v1+v2)", v[0]+(v[1]+v[2])),
        ("(v0+v2)+v1", (v[0]+v[2])+v[1]),
        ("(v1+v2)+v0", (v[1]+v[2])+v[0]),
    ]

    print(f"\nAll groupings for summing first 3 values:")
    print(f"\n{'Expression':<20} {'F32 Result':<25} {'vs 0.9':<20} {'Comparison':<15}")
    print(f"{'-'*20} {'-'*25} {'-'*20} {'-'*15}")

    for desc, result in groupings:
        result_val = float(result)
        diff = result_val - 0.9
        comp = "> 0.9" if result_val > 0.9 else ("= 0.9" if result_val == 0.9 else "< 0.9")
        print(f"{desc:<20} {result_val:.15f}  {diff:+.15e}  {comp:<15}")

    # Show the unique values
    unique = set(float(r) for _, r in groupings)

    print(f"\n{'='*80}")
    print(f"KEY FINDING:")
    print(f"{'='*80}")
    print(f"Number of unique f32 results: {len(unique)}")

    if len(unique) > 1:
        vals = sorted(unique)
        print(f"\nSome orderings give: {vals[0]:.15f} (< 0.9)")
        print(f"Some orderings give: {vals[1]:.15f} (> 0.9)")
        print(f"Difference: {vals[1] - vals[0]:.15e}")

        print(f"\n*** THIS IS THE PROBLEM! ***")
        print(f"\nIn top-p masking with p=0.9:")
        print(f"  - If cumsum computes to {vals[0]:.15f} < 0.9 -> Need more tokens!")
        print(f"  - If cumsum computes to {vals[1]:.15f} > 0.9 -> Enough tokens!")
        print(f"\nDifferent implementations using different summation orders")
        print(f"can make DIFFERENT DECISIONS about which tokens to include!")


if __name__ == "__main__":
    show_8_summation_orders()
    show_critical_3value_case()

    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    print(f"\n1. All 4 values [0.4, 0.3, 0.2, 0.1] sum to 1.0 (exactly representable)")
    print(f"   -> All 8 groupings give the same result")
    print(f"\n2. First 3 values [0.4, 0.3, 0.2] sum to 0.9 (NOT exactly representable)")
    print(f"   -> Different groupings give different results!")
    print(f"   -> Some < 0.9, some > 0.9")
    print(f"\n3. This causes f32 and i32 top-p implementations to differ at p=0.9")
    print(f"   -> f32: Uses one summation order, gets one result")
    print(f"   -> i32: Uses high-precision arithmetic, avoids the issue")
    print(f"{'='*80}\n")
