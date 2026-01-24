"""Demonstrate f32 vs i32 precision differences with a simple 4-value example.

With k=4 (all 4 values pass top-k), we focus on top-p masking where summation
order and precision differences can cause different outcomes.
"""

import jax
import jax.numpy as jnp
from tallax.vllm.topp_mask import topp_mask as i32_topp_mask
from tallax.vllm.tpu_inference_sampling_as_standalone_file import topp_mask as f32_topp_mask


def find_precision_sensitive_4values():
    """Find 4 probability values where f32 vs i32 summation gives different results."""

    print("="*70)
    print("Finding 4-value example where f32 vs i32 precision matters")
    print("="*70)

    # Strategy: Create 4 probabilities where:
    # 1. sum(p1, p2, p3) ≈ 0.95 but with rounding differences
    # 2. Choose top_p = 0.95 to fall right at the boundary
    # 3. The i32 quantization causes a different decision than f32

    scale = 2**30  # i32 scale factor used in i32_topp_mask

    # Start with target probabilities that sum to approximately 0.95
    # We want: p1 + p2 + p3 ≈ 0.95, p4 ≈ 0.05

    # Try probabilities that don't divide evenly when scaled to i32
    # Use fractions that have repeating binary representations

    # Example: 1/3 in binary is 0.010101... (repeating)
    # This will cause rounding differences

    target_sum_3 = 0.95
    target_sum_4 = 1.0

    # Try different probability distributions
    test_cases = [
        # Each probability as (approx value, description)
        ([0.316666667, 0.316666667, 0.316666666, 0.05], "Three ~1/3, one 0.05"),
        ([0.3167, 0.3167, 0.3166, 0.05], "Slightly varied thirds"),
        ([0.31, 0.32, 0.32, 0.05], "Varied values"),
        ([0.4, 0.3, 0.25, 0.05], "Descending"),
        ([0.24, 0.24, 0.24, 0.28], "Three 0.24, one 0.28"),
    ]

    for probs_unnorm, desc in test_cases:
        # Normalize to sum to 1.0
        probs = jnp.array(probs_unnorm, dtype=jnp.float32)
        probs = probs / probs.sum()

        print(f"\n{'='*70}")
        print(f"Testing: {desc}")
        print(f"{'='*70}")
        print(f"Probabilities: {probs}")
        print(f"Sum: {probs.sum():.15f}")

        # Compute cumulative sums
        cumsum_f32 = jnp.cumsum(probs)
        print(f"\nf32 cumulative sum:")
        for i, cs in enumerate(cumsum_f32):
            print(f"  After token {i}: {cs:.15f}")

        # Simulate i32 approach
        probs_i32_scaled = (probs * scale).astype(jnp.int32)
        cumsum_i32_scaled = jnp.cumsum(probs_i32_scaled)
        cumsum_i32_f32 = cumsum_i32_scaled.astype(jnp.float32) / scale

        print(f"\ni32 cumulative sum (after conversion back to f32):")
        for i, cs in enumerate(cumsum_i32_f32):
            print(f"  After token {i}: {cs:.15f}")

        # Check differences
        diff = cumsum_f32 - cumsum_i32_f32
        print(f"\nDifferences (f32 - i32):")
        for i, d in enumerate(diff):
            print(f"  After token {i}: {d:.15e}")

        # Find if there's a top_p value where they differ
        # Focus on the boundary between 2 and 3 tokens, and 3 and 4 tokens

        boundaries = [
            (2, cumsum_f32[1], cumsum_f32[2]),  # Between 2nd and 3rd token
            (3, cumsum_f32[2], cumsum_f32[3]),  # Between 3rd and 4th token
        ]

        for boundary_idx, lower, upper in boundaries:
            # Test top_p values in this range
            test_ps = jnp.linspace(float(lower) - 0.001, float(upper) + 0.001, 50)

            for top_p in test_ps:
                # Check how many tokens each approach would keep
                # f32: keep tokens where cumsum <= top_p
                f32_keep = (cumsum_f32 <= top_p).sum()
                i32_keep = (cumsum_i32_f32 <= top_p).sum()

                if f32_keep != i32_keep:
                    print(f"\n*** FOUND DIFFERENCE at boundary {boundary_idx}! ***")
                    print(f"top_p = {top_p:.15f}")
                    print(f"f32 would keep {f32_keep} tokens")
                    print(f"i32 would keep {i32_keep} tokens")
                    print(f"\nf32 cumsum: {cumsum_f32}")
                    print(f"i32 cumsum: {cumsum_i32_f32}")

                    # Now test with actual topp_mask functions
                    return verify_with_actual_functions(probs, top_p)

    print("\n" + "="*70)
    print("No difference found in test cases")
    print("="*70)
    return None


def verify_with_actual_functions(probs, top_p):
    """Verify the difference with actual topp_mask implementations."""

    print(f"\n{'='*70}")
    print("Verifying with actual topp_mask functions")
    print(f"{'='*70}")

    # Convert probabilities back to logits
    # logit = log(prob) + constant
    C = 10.0
    logits = jnp.log(probs) + C
    logits = logits.reshape(1, -1).astype(jnp.float32)

    print(f"\nLogits: {logits[0]}")

    # Check actual probabilities
    actual_probs = jax.nn.softmax(logits, axis=-1)[0]
    print(f"Actual probabilities from softmax: {actual_probs}")
    print(f"Sum: {actual_probs.sum()}")

    # Apply both topp_mask implementations
    i32_result = i32_topp_mask(logits, top_p, replace_val=-1e12)
    f32_result = f32_topp_mask(logits, top_p, replace_val=-1e12, stable=False)

    i32_mask = (i32_result[0] == -1e12)
    f32_mask = (f32_result[0] == -1e12)

    i32_kept = (~i32_mask).sum()
    f32_kept = (~f32_mask).sum()

    print(f"\nResults:")
    print(f"  i32 implementation: kept {i32_kept} tokens")
    print(f"  f32 implementation: kept {f32_kept} tokens")
    print(f"  i32 mask: {i32_mask}")
    print(f"  f32 mask: {f32_mask}")

    if i32_kept != f32_kept:
        print(f"\n{'='*70}")
        print("SUCCESS! Found example where implementations differ")
        print(f"{'='*70}")
        return logits, top_p, actual_probs
    else:
        print("\nNo difference in actual implementations (rounding absorbed by other operations)")
        return None


def construct_targeted_example():
    """Construct a targeted example where we know there will be a difference."""

    print("\n" + "="*70)
    print("Constructing targeted 4-value example")
    print("="*70)

    scale = 2**30

    # Choose probabilities carefully:
    # We want cumsum[2] to be very close to 0.95, but where
    # f32 and i32 rounding put it on opposite sides

    # Start by working backwards from i32
    # Choose i32 values that give us cumsum[2] just below 0.95

    # If cumsum[2] in i32 should be just below 0.95:
    # cumsum[2] * scale should be just below 0.95 * scale

    target_cumsum_2 = 0.95
    target_cumsum_2_scaled = int(target_cumsum_2 * scale)

    print(f"Target cumsum[2] (i32): {target_cumsum_2}")
    print(f"Target cumsum[2] scaled: {target_cumsum_2_scaled}")
    print(f"As f32: {target_cumsum_2_scaled / scale:.15f}")

    # Now choose 3 i32 values that sum to this
    # Use values that when converted to f32 probabilities, give different rounding

    # Example: divide roughly equally with intentional rounding
    val1_i32 = target_cumsum_2_scaled // 3
    val2_i32 = target_cumsum_2_scaled // 3
    val3_i32 = target_cumsum_2_scaled - val1_i32 - val2_i32  # Remainder

    print(f"\ni32 scaled values (first 3):")
    print(f"  val1: {val1_i32}")
    print(f"  val2: {val2_i32}")
    print(f"  val3: {val3_i32}")
    print(f"  sum:  {val1_i32 + val2_i32 + val3_i32}")

    # Convert to probabilities (these will be unnormalized)
    p1 = val1_i32 / scale
    p2 = val2_i32 / scale
    p3 = val3_i32 / scale

    # Choose p4 to make them sum to 1.0
    p4 = 1.0 - (p1 + p2 + p3)

    probs_unnorm = jnp.array([p1, p2, p3, p4], dtype=jnp.float32)

    print(f"\nUnnormalized probabilities: {probs_unnorm}")
    print(f"Sum: {probs_unnorm.sum():.15f}")

    # These should already be normalized, but let's check
    probs = probs_unnorm / probs_unnorm.sum()

    print(f"Normalized probabilities: {probs}")

    # Compute cumsums
    cumsum_f32 = jnp.cumsum(probs)
    probs_i32_scaled = (probs * scale).astype(jnp.int32)
    cumsum_i32_f32 = jnp.cumsum(probs_i32_scaled).astype(jnp.float32) / scale

    print(f"\nf32 cumsum: {cumsum_f32}")
    print(f"i32 cumsum: {cumsum_i32_f32}")
    print(f"Differences: {cumsum_f32 - cumsum_i32_f32}")

    # Choose top_p right at the boundary
    top_p = 0.95

    # Check which approach keeps how many tokens
    f32_keep = (cumsum_f32 <= top_p).sum()
    i32_keep = (cumsum_i32_f32 <= top_p).sum()

    print(f"\nWith top_p = {top_p}:")
    print(f"  f32 keeps: {f32_keep} tokens")
    print(f"  i32 keeps: {i32_keep} tokens")

    if f32_keep != i32_keep:
        print("\n*** DIFFERENCE FOUND! ***")
        return verify_with_actual_functions(probs, top_p)
    else:
        print("\nNo difference (need to adjust values)")
        return None


if __name__ == "__main__":
    print("Demonstrating f32 vs i32 precision with 4-value example")
    print("(k=4, so all values pass top-k, focusing on top-p)\n")

    # Try finding an example
    result = find_precision_sensitive_4values()

    if result is None:
        print("\nTrying targeted construction...")
        result = construct_targeted_example()

    if result is not None:
        logits, top_p, probs = result
        print(f"\n{'='*70}")
        print("FINAL EXAMPLE")
        print(f"{'='*70}")
        print(f"Logits: {logits[0]}")
        print(f"Probabilities: {probs}")
        print(f"top_p: {top_p}")
        print(f"\nThis example demonstrates how f32 summation order")
        print(f"can cause different masking results compared to i32 high-precision arithmetic.")
    else:
        print(f"\n{'='*70}")
        print("No example found - may need different value ranges")
        print(f"{'='*70}")
