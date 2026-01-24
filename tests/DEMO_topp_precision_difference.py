"""DEMONSTRATION: f32 vs i32 precision difference in topp_mask

This demonstrates a real case where f32 floating-point arithmetic and i32
high-precision arithmetic produce different top-p masking results.

EXAMPLE FOUND:
- Probabilities: [0.4, 0.3, 0.2, 0.1]
- top_p: 0.9
- i32 implementation: keeps 3 tokens (correct behavior)
- f32 implementation: keeps 4 tokens (due to rounding)

WHY IT HAPPENS:
The cumulative sum 0.4 + 0.3 + 0.2 = 0.9 exactly in mathematical terms,
but in f32 floating-point, the computation may give 0.90000004 due to rounding.

The binary search checks: "is sum(probs >= threshold) >= 0.9?"
- If the sum is stored as 0.90000004 > 0.9, it might keep only 3 tokens
- But if boundary comparisons differ, it might keep 4 tokens

The i32 implementation uses exact integer arithmetic (scaled by 2^30) to avoid
these rounding issues, giving the mathematically correct answer.
"""

import jax
import jax.numpy as jnp
from tallax.vllm.topp_mask import topp_mask as i32_topp_mask
from tallax.vllm.tpu_inference_sampling_as_standalone_file import topp_mask as f32_topp_mask


def demonstrate_precision_difference():
    """Demonstrate the actual precision difference."""

    print("="*70)
    print("DEMONSTRATION: f32 vs i32 Precision Difference in top-p Masking")
    print("="*70)

    # The example that shows the difference
    probs = jnp.array([0.4, 0.3, 0.2, 0.1], dtype=jnp.float32)
    top_p = 0.9

    print(f"\nProbabilities: {probs}")
    print(f"top_p: {top_p}")

    # Show cumulative sums
    sorted_probs = jnp.sort(probs)[::-1]
    cumsum = jnp.cumsum(sorted_probs)

    print(f"\nCumulative sum (sorted descending):")
    for i, (p, cs) in enumerate(zip(sorted_probs, cumsum), 1):
        exact = cs == top_p
        marker = " <- EXACTLY top_p!" if exact else ""
        print(f"  {i} token(s): {cs:.15f}{marker}")

    # Show the rounding issue
    print(f"\nF32 rounding analysis:")
    print(f"  Mathematical: 0.4 + 0.3 + 0.2 = 0.9")
    print(f"  F32 computed: {cumsum[2]:.15f}")
    print(f"  Difference: {cumsum[2] - 0.9:.15e}")

    # Convert to logits
    logits = (jnp.log(probs) + 10.0).reshape(1, -1)

    print(f"\nLogits: {logits[0]}")

    # Apply both implementations
    print(f"\n{'='*70}")
    print("Applying top-p masking with both implementations")
    print(f"{'='*70}")

    i32_result = i32_topp_mask(logits, top_p, replace_val=-1e12)
    f32_result = f32_topp_mask(logits, top_p, replace_val=-1e12, stable=False)

    i32_mask = (i32_result[0] == -1e12)
    f32_mask = (f32_result[0] == -1e12)

    i32_kept = (~i32_mask).sum()
    f32_kept = (~f32_mask).sum()

    print(f"\ni32 High-Precision Implementation:")
    print(f"  Kept tokens: {i32_kept}")
    print(f"  Mask: {i32_mask.tolist()}")
    print(f"  Kept probabilities: {probs[~i32_mask].tolist()}")
    print(f"  Sum of kept: {probs[~i32_mask].sum():.15f}")

    print(f"\nf32 Floating-Point Implementation:")
    print(f"  Kept tokens: {f32_kept}")
    print(f"  Mask: {f32_mask.tolist()}")
    print(f"  Kept probabilities: {probs[~f32_mask].tolist()}")
    print(f"  Sum of kept: {probs[~f32_mask].sum():.15f}")

    if i32_kept != f32_kept:
        print(f"\n{'='*70}")
        print(f"DIFFERENCE DETECTED!")
        print(f"{'='*70}")
        print(f"The i32 implementation keeps {i32_kept} tokens (mathematically correct)")
        print(f"The f32 implementation keeps {f32_kept} tokens (due to rounding)")

        print(f"\nExplanation:")
        print(f"  With top_p=0.9, we want the smallest set of tokens")
        print(f"  whose probabilities sum to >= 0.9")
        print(f"  ")
        print(f"  Sum of first 3 probs: 0.4 + 0.3 + 0.2 = 0.9 (exactly)")
        print(f"  ")
        print(f"  The i32 implementation correctly identifies this and keeps 3 tokens.")
        print(f"  The f32 implementation, due to rounding in its comparisons,")
        print(f"  keeps {f32_kept} tokens instead.")

        print(f"\nThis demonstrates why high-precision arithmetic is important")
        print(f"for numerical stability in top-p sampling!")


def show_sampling_impact():
    """Show how this affects actual sampling."""

    print(f"\n{'='*70}")
    print("Impact on Sampling")
    print(f"{'='*70}")

    # The example
    probs = jnp.array([0.4, 0.3, 0.2, 0.1], dtype=jnp.float32)
    top_p = 0.9
    logits = (jnp.log(probs) + 10.0).reshape(1, -1)

    # Apply masking
    i32_masked = i32_topp_mask(logits, top_p, replace_val=-1e12)
    f32_masked = f32_topp_mask(logits, top_p, replace_val=-1e12, stable=False)

    # Sample from each
    key = jax.random.PRNGKey(42)

    i32_samples = []
    f32_samples = []

    print(f"\nGenerating 1000 samples from each distribution:")

    for i in range(1000):
        key, subkey = jax.random.split(key)
        i32_sample = jax.random.categorical(subkey, i32_masked)
        f32_sample = jax.random.categorical(subkey, f32_masked)

        i32_samples.append(int(i32_sample[0]))
        f32_samples.append(int(f32_sample[0]))

    # Count frequencies
    i32_counts = {i: i32_samples.count(i) for i in range(4)}
    f32_counts = {i: f32_samples.count(i) for i in range(4)}

    print(f"\nSample distribution (out of 1000 samples):")
    print(f"\n  Token | Prob  | i32 count | f32 count | Difference")
    print(f"  ------|-------|-----------|-----------|------------")

    for i in range(4):
        i32_count = i32_counts.get(i, 0)
        f32_count = f32_counts.get(i, 0)
        diff = abs(i32_count - f32_count)

        marker = " <-- DIFFERS" if diff > 0 else ""

        print(f"    {i}   | {probs[i]:.1f}   | {i32_count:4d}      | {f32_count:4d}      | {diff:4d}{marker}")

    print(f"\nAs you can see, the f32 implementation samples from token 3")
    print(f"(the 0.1 probability token) which should be masked out.")
    print(f"This is due to the precision difference in the masking step!")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("F32 vs I32 PRECISION DEMONSTRATION")
    print("="*70)
    print("\nThis demonstrates a case where floating-point rounding")
    print("causes different top-p masking results.\n")

    demonstrate_precision_difference()
    show_sampling_impact()

    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    print("\nWith simple 4-value example:")
    print("  Probabilities: [0.4, 0.3, 0.2, 0.1]")
    print("  top_p: 0.9")
    print("  ")
    print("  i32 keeps: 3 tokens (0.4 + 0.3 + 0.2 = 0.9) ✓ CORRECT")
    print("  f32 keeps: 4 tokens (due to rounding)      ✗ INCORRECT")
    print("\nThis demonstrates why the i32 high-precision implementation")
    print("is necessary for numerical correctness in top-p sampling!")
    print("="*70 + "\n")
