"""Create 4-value example where summation order causes f32 vs i32 to differ.

The key: f32_topp_mask uses jnp.sum (unspecified order, f32 accumulation)
         i32_topp_mask uses high-precision int32 accumulation

We need probabilities where the sum in f32 differs from sum in i32 enough to
change whether we cross the top_p threshold.
"""

import jax
import jax.numpy as jnp
import numpy as np
from tallax.vllm.topp_mask import topp_mask as i32_topp_mask
from tallax.vllm.tpu_inference_sampling_as_standalone_file import topp_mask as f32_topp_mask


def test_summation_precision():
    """Test where f32 sum differs from i32 high-precision sum."""

    print("="*70)
    print("Demonstrating summation precision differences")
    print("="*70)

    # Create 4 probabilities where:
    # - Their sum should be close to top_p=0.95
    # - But f32 rounding causes the sum to be slightly different than i32

    # Use probabilities that don't have exact f32 representations
    # Focus on values where catastrophic cancellation or precision loss occurs

    # Strategy: Use one large value and several small values
    # where small + small + small in f32 loses precision

    # In f32, we have ~7 decimal digits of precision
    # If we have: large = 0.9499999, small1 = 0.000000033, small2 = 0.000000033, small3 = 0.000000034
    # The sum might lose precision

    vocab_size = 4

    # Test various configurations
    test_cases = [
        {
            'name': 'One dominant, three tiny',
            'probs_unnorm': [0.949999999, 1e-8, 1e-8, 0.05],
            'top_p': 0.95,
        },
        {
            'name': 'Values at f32 precision limit',
            'probs_unnorm': [0.316666667, 0.316666667, 0.316666666, 0.05],
            'top_p': 0.95,
        },
        {
            'name': 'Precision-sensitive distribution',
            'probs_unnorm': [0.475, 0.475, 1e-7, 1e-7],
            'top_p': 0.95,
        },
    ]

    for test_case in test_cases:
        print(f"\n{'='*70}")
        print(f"Test: {test_case['name']}")
        print(f"{'='*70}")

        probs_unnorm = jnp.array(test_case['probs_unnorm'], dtype=jnp.float32)
        probs = probs_unnorm / probs_unnorm.sum()
        top_p = test_case['top_p']

        print(f"Probabilities: {probs}")
        print(f"Sum: {probs.sum():.15f}")

        # Convert to logits
        C = 10.0
        logits = jnp.log(probs) + C
        logits = logits.reshape(1, -1)

        # Apply both implementations
        i32_result = i32_topp_mask(logits, top_p, replace_val=-1e12)
        f32_result = f32_topp_mask(logits, top_p, replace_val=-1e12, stable=False)

        i32_mask = (i32_result[0] == -1e12)
        f32_mask = (f32_result[0] == -1e12)

        i32_kept = (~i32_mask).sum()
        f32_kept = (~f32_mask).sum()

        print(f"\nResults with top_p={top_p}:")
        print(f"  i32: kept {i32_kept} tokens, mask={i32_mask}")
        print(f"  f32: kept {f32_kept} tokens, mask={f32_mask}")

        if i32_kept != f32_kept:
            print(f"\n*** FOUND DIFFERENCE! ***")
            return logits, top_p, probs


def manual_precision_construction():
    """Manually construct a case where we know f32 and i32 will differ."""

    print("\n" + "="*70)
    print("Manual construction of precision-sensitive example")
    print("="*70)

    # The i32 scale factor
    scale = 2**30

    # Create probabilities by working directly with i32 values
    # Choose i32 values where:
    # 1. sum of 3 values in i32 is just below 0.95 * scale
    # 2. When converted to f32, rounding puts the sum just above 0.95

    # Example: Make 3 equal values
    target_for_3 = 0.95
    i32_for_3 = int(target_for_3 * scale)

    # Split into 3 equal parts (with remainder)
    i32_val1 = i32_for_3 // 3
    i32_val2 = i32_for_3 // 3
    i32_val3 = i32_for_3 - i32_val1 - i32_val2

    print(f"\ni32 values (first 3):")
    print(f"  val1: {i32_val1}")
    print(f"  val2: {i32_val2}")
    print(f"  val3: {i32_val3}")
    print(f"  sum:  {i32_val1 + i32_val2 + i32_val3}")
    print(f"  sum as fraction of scale: {(i32_val1 + i32_val2 + i32_val3) / scale}")

    # Convert to f32
    p1_f32 = i32_val1 / scale
    p2_f32 = i32_val2 / scale
    p3_f32 = i32_val3 / scale

    # Check what happens when we sum in f32
    sum_f32_sequential = p1_f32 + p2_f32 + p3_f32
    sum_f32_pairwise = (p1_f32 + p2_f32) + p3_f32

    print(f"\nf32 values:")
    print(f"  p1: {p1_f32:.15f}")
    print(f"  p2: {p2_f32:.15f}")
    print(f"  p3: {p3_f32:.15f}")
    print(f"  sum (sequential): {sum_f32_sequential:.15f}")
    print(f"  sum (pairwise):   {sum_f32_pairwise:.15f}")

    # Back to i32 and sum
    sum_i32 = int(p1_f32 * scale) + int(p2_f32 * scale) + int(p3_f32 * scale)
    sum_i32_as_f32 = sum_i32 / scale

    print(f"  sum via i32 round-trip: {sum_i32_as_f32:.15f}")

    # The difference
    print(f"\nDifference (f32 - i32): {sum_f32_sequential - sum_i32_as_f32:.15e}")

    # Now create full probability distribution
    # Add a 4th value to make it sum to 1.0
    p4 = max(0.0, 1.0 - sum_f32_sequential)

    probs = jnp.array([p1_f32, p2_f32, p3_f32, p4], dtype=jnp.float32)
    probs = probs / probs.sum()  # Normalize

    print(f"\nFull probability distribution: {probs}")

    # Test with different top_p values
    for top_p in [0.949, 0.9499, 0.95, 0.9501, 0.951]:
        print(f"\nTesting top_p={top_p:.4f}:")

        # Convert to logits
        logits = jnp.log(probs) + 10.0
        logits = logits.reshape(1, -1)

        i32_result = i32_topp_mask(logits, top_p, replace_val=-1e12)
        f32_result = f32_topp_mask(logits, top_p, replace_val=-1e12, stable=False)

        i32_kept = (i32_result[0] != -1e12).sum()
        f32_kept = (f32_result[0] != -1e12).sum()

        print(f"  i32: {i32_kept} tokens, f32: {f32_kept} tokens", end="")

        if i32_kept != f32_kept:
            print(" *** DIFFERENT! ***")
            return logits, top_p, probs
        else:
            print()

    return None, None, None


def brute_force_search():
    """Brute force search for a case where they differ."""

    print("\n" + "="*70)
    print("Brute force search for precision difference")
    print("="*70)

    # Try many random 4-value distributions
    key = jax.random.PRNGKey(42)

    for i in range(100):
        key, subkey = jax.random.split(key)

        # Generate random unnormalized probabilities
        unnorm_probs = jax.random.uniform(subkey, (4,), minval=0.01, maxval=1.0)
        probs = unnorm_probs / unnorm_probs.sum()

        # Convert to logits
        logits = jnp.log(probs) + 10.0
        logits = logits.reshape(1, -1)

        # Test various top_p values
        for top_p in jnp.linspace(0.1, 0.99, 50):
            i32_result = i32_topp_mask(logits, float(top_p), replace_val=-1e12)
            f32_result = f32_topp_mask(logits, float(top_p), replace_val=-1e12, stable=False)

            i32_kept = (i32_result[0] != -1e12).sum()
            f32_kept = (f32_result[0] != -1e12).sum()

            if i32_kept != f32_kept:
                print(f"\n*** FOUND at iteration {i}! ***")
                print(f"Probabilities: {probs}")
                print(f"top_p: {top_p}")
                print(f"i32 kept: {i32_kept}, f32 kept: {f32_kept}")
                return logits, float(top_p), probs

        if (i + 1) % 10 == 0:
            print(f"Tested {i+1} distributions...")

    print("\nNo difference found in brute force search")
    return None, None, None


if __name__ == "__main__":
    print("Searching for 4-value example where f32 vs i32 summation differs\n")

    # Try different approaches
    result = test_summation_precision()
    if result is None:
        result = manual_precision_construction()
    if result is None:
        result = brute_force_search()

    if result[0] is not None:
        logits, top_p, probs = result
        print(f"\n{'='*70}")
        print("SUCCESS! Found example")
        print(f"{'='*70}")
        print(f"Logits: {logits[0]}")
        print(f"Probabilities: {probs}")
        print(f"top_p: {top_p}")
    else:
        print(f"\n{'='*70}")
        print("No example found")
        print(f"{'='*70}")
        print("\nThis suggests the i32 high-precision implementation is working")
        print("correctly to avoid the precision issues that would occur in pure f32!")
