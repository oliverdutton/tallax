"""Demonstrate f32 vs i32 precision differences in topp_mask implementations.

This test shows how floating-point summation order can cause different results
between the f32 topp_mask (standalone) and i32 high-precision topp_mask (Pallas).
"""

import jax
import jax.numpy as jnp
from tallax.vllm.topp_mask import topp_mask as i32_topp_mask
from tallax.vllm.tpu_inference_sampling_as_standalone_file import topp_mask as f32_topp_mask


def test_f32_summation_order_matters():
    """Demonstrate that f32 summation order can affect results."""

    # Create a simple example where f32 summation order matters
    # Use three values that sum to ~0.95 but have rounding differences

    # Strategy: Create logits where probabilities are approximately p1 ≈ p2 ≈ p3 ≈ 0.95/3 ≈ 0.3167
    # but with specific values chosen so that:
    # - sum(p1, p2) + p3 ≠ sum(p2, p3) + p1 in f32
    # - And choose top_p to fall right at the boundary

    print("="*70)
    print("Testing f32 summation order sensitivity")
    print("="*70)

    # Start with equal logits
    base_logit = 0.0

    # Create three logits with small differences
    # After softmax, these should be approximately equal but with rounding issues
    logits = jnp.array([base_logit, base_logit + 1e-7, base_logit - 1e-7], dtype=jnp.float32)

    # Add more elements to make it realistic
    logits = jnp.pad(logits, (0, 1021), constant_values=-10.0)
    logits = logits.reshape(1, -1)

    print(f"\nLogits (first 3 values): {logits[0, :3]}")

    # Compute probabilities
    probs = jax.nn.softmax(logits, axis=-1)[0]
    print(f"Probabilities (first 3 values): {probs[:3]}")
    print(f"Sum of first 3 probs: {probs[:3].sum()}")

    # Test different top_p values around the boundary
    for top_p in [0.95, 0.949, 0.951, 0.9499, 0.9501]:
        print(f"\n{'='*70}")
        print(f"Testing with top_p = {top_p}")
        print(f"{'='*70}")

        i32_result = i32_topp_mask(logits, top_p, replace_val=-1e12)
        f32_result = f32_topp_mask(logits, top_p, replace_val=-1e12, stable=False)

        i32_mask = (i32_result == -1e12)
        f32_mask = (f32_result == -1e12)

        i32_kept = (~i32_mask).sum()
        f32_kept = (~f32_mask).sum()

        print(f"  i32 implementation kept: {i32_kept} tokens")
        print(f"  f32 implementation kept: {f32_kept} tokens")

        if i32_kept != f32_kept:
            print(f"  *** DIFFERENCE FOUND! ***")
            diff_indices = jnp.where(i32_mask[0] != f32_mask[0])[0]
            print(f"  Differing indices: {diff_indices}")
            return True

    return False


def create_precision_sensitive_example():
    """Create a minimal example where f32 rounding causes different topp_mask results."""

    print("\n" + "="*70)
    print("Creating precision-sensitive example")
    print("="*70)

    # Use the classic example: values that demonstrate f32 associativity issues
    # Based on the fact that (a + b) + c ≠ a + (b + c) in floating point

    # Find three values where order matters
    # Use values near the f32 precision limit
    v1 = 1.0
    v2 = 1e-7
    v3 = -1e-7

    # Test associativity
    sum1 = (v1 + v2) + v3  # Left-to-right
    sum2 = v1 + (v2 + v3)  # Right-to-left
    sum3 = (v1 + v3) + v2  # Different order

    print(f"\nDemonstrating f32 associativity issues:")
    print(f"  v1={v1}, v2={v2}, v3={v3}")
    print(f"  (v1 + v2) + v3 = {sum1}")
    print(f"  v1 + (v2 + v3) = {sum2}")
    print(f"  (v1 + v3) + v2 = {sum3}")
    print(f"  Difference: {abs(sum1 - sum2)}")

    # Now create logits that will expose this in topp_mask
    # We want probabilities where the cumulative sum order matters

    # Strategy: Create a case where we have many small equal values
    # and the cumulative sum crosses the threshold differently depending on order

    vocab_size = 1024

    # Create logits that give us controlled probabilities
    # Use a few large values and many small equal values

    # Main probability mass (3 values each with ~30% prob)
    main_logits = jnp.array([10.0, 10.0 + 1e-6, 10.0 - 1e-6])

    # Rest with very small probability
    rest_logits = jnp.full(vocab_size - 3, 0.0)

    logits = jnp.concatenate([main_logits, rest_logits]).reshape(1, -1).astype(jnp.float32)

    # Compute actual probabilities
    probs = jax.nn.softmax(logits, axis=-1)[0]

    print(f"\n3 main probabilities: {probs[:3]}")
    print(f"Sum of 3 main probs: {probs[:3].sum():.15f}")
    print(f"Sum of 2 first probs: {probs[:2].sum():.15f}")
    print(f"Sum of 2 last probs: {probs[1:3].sum():.15f}")

    # Find a top_p that falls between 2 and 3 tokens
    two_token_sum = probs[:2].sum()
    three_token_sum = probs[:3].sum()

    print(f"\nCumulative probability analysis:")
    print(f"  2 tokens: {two_token_sum:.15f}")
    print(f"  3 tokens: {three_token_sum:.15f}")

    # Test with top_p values in between
    for top_p in jnp.linspace(two_token_sum - 0.001, three_token_sum + 0.001, 20):
        i32_result = i32_topp_mask(logits, float(top_p), replace_val=-1e12)
        f32_result = f32_topp_mask(logits, float(top_p), replace_val=-1e12, stable=False)

        i32_kept = ((i32_result != -1e12).sum())
        f32_kept = ((f32_result != -1e12).sum())

        if i32_kept != f32_kept:
            print(f"\n*** FOUND DIFFERENCE at top_p={top_p:.15f} ***")
            print(f"  i32 kept: {i32_kept} tokens")
            print(f"  f32 kept: {f32_kept} tokens")
            return logits, float(top_p)

    print("\nNo difference found with this example")
    return None, None


def create_extreme_precision_example():
    """Create an extreme example with carefully crafted probabilities."""

    print("\n" + "="*70)
    print("Creating extreme precision-sensitive example")
    print("="*70)

    # Use a more extreme approach: Create probabilities where the i32 scaling
    # introduces quantization that changes the cumulative sum threshold

    # With i32 scaling to 2^30, we get precision of ~1/2^30 ≈ 1e-9
    # But f32 has precision of ~1e-7, so there's a gap

    scale = 2**30

    # Create three probabilities that are affected by i32 quantization
    # Each is approximately 1/3, but with specific f32 values that when
    # converted to i32 and back, give slightly different sums

    # Start with a probability that doesn't divide evenly in i32
    p_approx_third = 1.0 / 3.0

    # When scaled to i32 and back:
    p_i32_quantized = int(p_approx_third * scale) / scale

    print(f"\nProbability quantization:")
    print(f"  f32: 1/3 = {p_approx_third:.15f}")
    print(f"  i32 scaled: {int(p_approx_third * scale)}")
    print(f"  i32 back to f32: {p_i32_quantized:.15f}")
    print(f"  Difference: {abs(p_approx_third - p_i32_quantized):.15e}")

    # Create logits that give us these probabilities
    # Use the inverse softmax formula: logit = log(prob) + C

    # Three nearly equal probabilities that sum to slightly less than 0.95
    # This way top_p=0.95 should include all 3, but rounding might change that

    target_probs = jnp.array([0.316, 0.317, 0.317])  # Sum = 0.950

    # Convert to logits (with arbitrary constant)
    C = 10.0
    main_logits = jnp.log(target_probs) + C

    # Rest with tiny probability
    vocab_size = 1024
    rest_logits = jnp.full(vocab_size - 3, -100.0)

    logits = jnp.concatenate([main_logits, rest_logits]).reshape(1, -1).astype(jnp.float32)

    # Check actual probabilities
    actual_probs = jax.nn.softmax(logits, axis=-1)[0]

    print(f"\nActual probabilities:")
    print(f"  First 3: {actual_probs[:3]}")
    print(f"  Sum of first 3: {actual_probs[:3].sum():.15f}")

    # Test around 0.95
    test_values = [0.949, 0.9499, 0.95, 0.9501, 0.951]

    for top_p in test_values:
        i32_result = i32_topp_mask(logits, top_p, replace_val=-1e12)
        f32_result = f32_topp_mask(logits, top_p, replace_val=-1e12, stable=False)

        i32_kept = (i32_result != -1e12).sum()
        f32_kept = (f32_result != -1e12).sum()

        print(f"\ntop_p={top_p:.4f}:")
        print(f"  i32: {i32_kept} tokens, f32: {f32_kept} tokens", end="")

        if i32_kept != f32_kept:
            print(f" *** DIFFERENCE! ***")
            return logits, top_p
        else:
            print()

    return None, None


def analyze_summation_approaches():
    """Analyze how the two approaches differ in their summation."""

    print("\n" + "="*70)
    print("Analyzing summation approaches")
    print("="*70)

    # Create a simple example with known probabilities
    probs = jnp.array([0.4, 0.3, 0.2, 0.09, 0.01], dtype=jnp.float32)

    print(f"\nTest probabilities: {probs}")
    print(f"Sum: {probs.sum():.15f}")

    # Simulate f32 cumsum (what f32_topp_mask does)
    f32_cumsum = jnp.cumsum(probs)
    print(f"\nf32 cumulative sum: {f32_cumsum}")

    # Simulate i32 approach (what i32_topp_mask does)
    scale = 2**30
    i32_scaled = (probs * scale).astype(jnp.int32)
    print(f"\ni32 scaled values: {i32_scaled}")

    # Sum in i32
    i32_cumsum_scaled = jnp.cumsum(i32_scaled)
    print(f"i32 cumulative sum (scaled): {i32_cumsum_scaled}")

    # Convert back to f32
    i32_cumsum_f32 = i32_cumsum_scaled.astype(jnp.float32) / scale
    print(f"i32 cumsum converted to f32: {i32_cumsum_f32}")

    # Show differences
    diff = f32_cumsum - i32_cumsum_f32
    print(f"\nDifference (f32 - i32): {diff}")
    print(f"Max absolute difference: {jnp.abs(diff).max():.15e}")

    # Now test with top_p values at boundaries
    for i, (top_p, cum_val) in enumerate(zip([0.39, 0.40, 0.41], f32_cumsum[:3])):
        print(f"\n  Testing top_p={top_p}:")
        print(f"    f32 cumsum[{i}] = {f32_cumsum[i]:.15f}")
        print(f"    i32 cumsum[{i}] = {i32_cumsum_f32[i]:.15f}")

        # Check which would include this token
        f32_includes = f32_cumsum[i] <= top_p
        i32_includes = i32_cumsum_f32[i] <= top_p

        if f32_includes != i32_includes:
            print(f"    *** DIFFERENT DECISIONS! f32={f32_includes}, i32={i32_includes} ***")


if __name__ == "__main__":
    print("Demonstrating f32 vs i32 precision differences in topp_mask\n")

    # Analyze the fundamental difference
    analyze_summation_approaches()

    # Try to find examples
    print("\n" + "="*70)
    print("Attempting to find real examples with different mask outcomes")
    print("="*70)

    test_f32_summation_order_matters()

    logits1, top_p1 = create_precision_sensitive_example()
    if logits1 is not None:
        print(f"\n✓ Found example 1: logits shape={logits1.shape}, top_p={top_p1}")

    logits2, top_p2 = create_extreme_precision_example()
    if logits2 is not None:
        print(f"\n✓ Found example 2: logits shape={logits2.shape}, top_p={top_p2}")
