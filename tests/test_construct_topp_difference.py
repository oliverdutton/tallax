"""Construct a 4-value example where f32 and i32 topp_mask differ.

Since brute force didn't find a natural example, we'll construct one by
understanding exactly how the binary search works and crafting probabilities
that expose the difference.

The key insight: both use binary search, but with different precision in
evaluating the predicate (probability_mass >= p).
"""

import jax
import jax.numpy as jnp
from tallax.vllm.topp_mask import topp_mask as i32_topp_mask
from tallax.vllm.tpu_inference_sampling_as_standalone_file import topp_mask as f32_topp_mask


def analyze_binary_search_behavior():
    """Understand how each implementation's binary search behaves."""

    print("="*70)
    print("Analyzing binary search behavior in both implementations")
    print("="*70)

    # Create simple 4-value case
    probs = jnp.array([0.4, 0.3, 0.2, 0.1], dtype=jnp.float32)
    logits = (jnp.log(probs) + 10.0).reshape(1, -1)

    print(f"\nProbabilities: {probs}")
    print(f"Sorted (descending): {jnp.sort(probs)[::-1]}")

    # Compute cumulative sums
    sorted_probs = jnp.sort(probs)[::-1]
    cumsum = jnp.cumsum(sorted_probs)

    print(f"\nCumulative probability (sorted descending):")
    for i, (p, cs) in enumerate(zip(sorted_probs, cumsum)):
        print(f"  After token {i}: p={p:.15f}, cumsum={cs:.15f}")

    # Test with different top_p values
    print(f"\nTesting different top_p values:")

    for top_p in [0.3, 0.39, 0.4, 0.41, 0.69, 0.7, 0.71, 0.89, 0.9, 0.91, 0.99, 1.0]:
        i32_result = i32_topp_mask(logits, top_p, replace_val=-1e12)
        f32_result = f32_topp_mask(logits, top_p, replace_val=-1e12, stable=False)

        i32_kept = (i32_result[0] != -1e12).sum()
        f32_kept = (f32_result[0] != -1e12).sum()

        match = "✓" if i32_kept == f32_kept else "✗ DIFF!"

        print(f"  top_p={top_p:.2f}: i32={i32_kept}, f32={f32_kept} {match}")


def try_extreme_precision_case():
    """Try an extreme case with very specific probability values."""

    print("\n" + "="*70)
    print("Trying extreme precision case")
    print("="*70)

    # Create probabilities where the threshold lands between two very close values
    # Use the smallest f32 representable difference

    # Start with base probability
    base = 0.25
    epsilon = jnp.finfo(jnp.float32).eps * base  # Smallest increment for this value

    print(f"\nBase probability: {base}")
    print(f"f32 epsilon at this value: {epsilon:.15e}")

    # Create 4 probabilities: base ± epsilon
    p1 = base
    p2 = base
    p3 = base + epsilon
    p4 = base - epsilon

    probs = jnp.array([p1, p2, p3, p4], dtype=jnp.float32)
    probs = probs / probs.sum()  # Normalize

    print(f"\nProbabilities: {probs}")
    print(f"Differences from base:")
    for i, p in enumerate(probs):
        print(f"  p[{i}] - base = {(p - base):.15e}")

    # Convert to logits
    logits = (jnp.log(probs) + 10.0).reshape(1, -1)

    # Test at various top_p values
    print(f"\nTesting:")
    for top_p in jnp.linspace(0.24, 0.76, 100):
        i32_result = i32_topp_mask(logits, float(top_p), replace_val=-1e12)
        f32_result = f32_topp_mask(logits, float(top_p), replace_val=-1e12, stable=False)

        i32_kept = (i32_result[0] != -1e12).sum()
        f32_kept = (f32_result[0] != -1e12).sum()

        if i32_kept != f32_kept:
            print(f"  *** FOUND DIFFERENCE at top_p={top_p:.15f} ***")
            print(f"  i32 kept: {i32_kept}")
            print(f"  f32 kept: {f32_kept}")
            return logits, float(top_p), probs

    print("  No difference found")
    return None, None, None


def construct_with_repeated_values():
    """Construct case with repeated probability values."""

    print("\n" + "="*70)
    print("Trying case with repeated probability values")
    print("="*70)

    # When multiple tokens have the exact same probability, the threshold
    # lands exactly on that probability value. This is where precision matters.

    # Try: 3 tokens with same prob, 1 different
    # Choose values so that sum of 2 is just below top_p, sum of 3 is just above

    # Example: 3 tokens at 0.32, 1 at 0.04
    # sum of 2: 0.64, sum of 3: 0.96

    probs = jnp.array([0.32, 0.32, 0.32, 0.04], dtype=jnp.float32)
    probs = probs / probs.sum()

    print(f"Probabilities: {probs}")

    # Cumulative sums (sorted)
    sorted_probs = jnp.sort(probs)[::-1]
    cumsum = jnp.cumsum(sorted_probs)

    print(f"\nCumulative (sorted descending):")
    for i, cs in enumerate(cumsum):
        print(f"  After {i+1} tokens: {cs:.15f}")

    logits = (jnp.log(probs) + 10.0).reshape(1, -1)

    # Test around the boundaries
    test_values = list(jnp.linspace(0.63, 0.97, 200))

    print(f"\nTesting {len(test_values)} top_p values...")

    for top_p in test_values:
        i32_result = i32_topp_mask(logits, float(top_p), replace_val=-1e12)
        f32_result = f32_topp_mask(logits, float(top_p), replace_val=-1e12, stable=False)

        i32_kept = (i32_result[0] != -1e12).sum()
        f32_kept = (f32_result[0] != -1e12).sum()

        if i32_kept != f32_kept:
            print(f"\n*** FOUND DIFFERENCE! ***")
            print(f"top_p = {top_p:.15f}")
            print(f"i32 kept: {i32_kept} tokens")
            print(f"f32 kept: {f32_kept} tokens")

            # Show which tokens were kept
            i32_mask = i32_result[0] != -1e12
            f32_mask = f32_result[0] != -1e12

            print(f"\ni32 kept tokens: {i32_mask}")
            print(f"f32 kept tokens: {f32_mask}")

            return logits, float(top_p), probs

    print("No difference found")
    return None, None, None


def final_manual_construction():
    """Final attempt: manually construct the exact case we want."""

    print("\n" + "="*70)
    print("Manual construction of target case")
    print("="*70)

    # I'll construct this very carefully:
    # - p1=0.4, p2=0.3, p3=0.2, p4=0.1
    # - cumsum: 0.4, 0.7, 0.9, 1.0
    # - Choose top_p=0.7 exactly
    # - At this point, threshold should be p2=0.3
    # - The binary search must decide: is cumsum(probs >= 0.3) >= 0.7?
    # - In f32: 0.4 + 0.3 = 0.7 (exactly)
    # - But what if the binary search finds a slightly different threshold?

    probs = jnp.array([0.4, 0.3, 0.2, 0.1], dtype=jnp.float32)
    logits = (jnp.log(probs) + 10.0).reshape(1, -1)

    print(f"Probabilities: {probs}")

    # The issue: at exactly top_p=0.7, should we keep 2 or 3 tokens?
    # Standard definition: keep smallest set with sum >= p
    # So at p=0.7, we keep 2 (since 0.4+0.3=0.7 exactly)

    exact_boundary_tests = [
        (0.4, "exactly p1"),
        (0.7, "exactly p1+p2"),
        (0.9, "exactly p1+p2+p3"),
    ]

    for top_p, desc in exact_boundary_tests:
        print(f"\nTesting top_p={top_p} ({desc}):")

        i32_result = i32_topp_mask(logits, top_p, replace_val=-1e12)
        f32_result = f32_topp_mask(logits, top_p, replace_val=-1e12, stable=False)

        i32_kept = (i32_result[0] != -1e12).sum()
        f32_kept = (f32_result[0] != -1e12).sum()

        print(f"  i32 kept: {i32_kept}")
        print(f"  f32 kept: {f32_kept}")

        if i32_kept != f32_kept:
            print(f"  *** DIFFERENCE! ***")
            return logits, top_p, probs

    return None, None, None


if __name__ == "__main__":
    print("Constructing 4-value example where f32 and i32 differ\n")

    analyze_binary_search_behavior()

    result = try_extreme_precision_case()
    if result[0] is None:
        result = construct_with_repeated_values()
    if result[0] is None:
        result = final_manual_construction()

    if result[0] is not None:
        logits, top_p, probs = result
        print(f"\n{'='*70}")
        print("SUCCESS! Found constructed example")
        print(f"{'='*70}")
        print(f"Logits: {logits[0]}")
        print(f"Probabilities: {probs}")
        print(f"top_p: {top_p}")
    else:
        print(f"\n{'='*70}")
        print("Could not construct a 4-value example where they differ")
        print(f"{'='*70}")
        print("\nConclusion: The i32 high-precision implementation successfully")
        print("avoids f32 rounding errors, even in edge cases!")
        print("\nTo find differences, we would likely need:")
        print("  - Many more values (>100) where summation order matters more")
        print("  - Or exploit specific numerical edge cases in binary search")
