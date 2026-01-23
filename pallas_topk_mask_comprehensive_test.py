"""Comprehensive tests for topk_mask_pallas with larger shapes and ties."""

import jax
import jax.numpy as jnp
from tallax.tax.pallas_topk_mask import topk_mask_pallas


def compare_with_jax_topk(x, k, test_name, verbose=True):
    """Compare our dense masked implementation with sparse jax.lax.top_k"""
    if verbose:
        print(f"\n{test_name}")
        print(f"  Shape: {x.shape}, k: {k}")

    # Our implementation (dense masked)
    our_result = topk_mask_pallas(x, k, replace_val=-jnp.inf, stable=True, interpret=True)

    topk_vals, topk_idxs = jax.lax.top_k(x, k)
    base = jnp.full_like(our_result, -jnp.inf)
    ref_result = jax.vmap(lambda x, idx, val: x.at[idx].set(val))(base, topk_idxs, topk_vals)
    return (our_result==ref_result).all()


def test_larger_shapes():
    """Test with larger shapes."""
    print("="*70)
    print("Test Suite: Larger Shapes")
    print("="*70)

    rng = jax.random.PRNGKey(42)

    # Test 1: (8, 128)
    x1 = jax.random.uniform(rng, (8, 128))
    assert compare_with_jax_topk(x1, 20, "Test 1: (8, 128) random uniform")

    # Test 2: (16, 2048)
    rng, subkey = jax.random.split(rng)
    x2 = jax.random.uniform(subkey, (16, 2048))
    assert compare_with_jax_topk(x2, 50, "Test 2: (16, 2048) random uniform")

    # Test 3: (32, 512)
    rng, subkey = jax.random.split(rng)
    x3 = jax.random.uniform(subkey, (32, 512))
    assert compare_with_jax_topk(x3, 30, "Test 3: (32, 512) random uniform")

    print("\n✅ All larger shape tests passed!\n")


def test_with_ties():
    """Test with various tie patterns."""
    print("="*70)
    print("Test Suite: Ties")
    print("="*70)

    rng = jax.random.PRNGKey(100)

    # Test 1: Moderate ties
    print("\nTest 1: (8, 128) with moderate ties")
    x1 = jax.random.uniform(rng, (8, 128))
    x1 = jnp.round(x1 * 5) / 5  # Values: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    print(f"  Unique values per batch: {[len(jnp.unique(x1[i])) for i in range(min(3, x1.shape[0]))]}")
    assert compare_with_jax_topk(x1, 20, "Moderate ties", verbose=False)
    print(f"  ✅ Pass")

    # Test 2: Many ties
    print("\nTest 2: (16, 2048) with many ties")
    rng, subkey = jax.random.split(rng)
    x2 = jax.random.uniform(subkey, (16, 2048))
    x2 = jnp.round(x2 * 10) / 10
    print(f"  Unique values per batch: {[len(jnp.unique(x2[i])) for i in range(min(3, x2.shape[0]))]}")
    assert compare_with_jax_topk(x2, 100, "Many ties", verbose=False)
    print(f"  ✅ Pass")

    # Test 3: Extreme ties (most values are the same)
    print("\nTest 3: (8, 128) with extreme ties")
    x3 = jnp.ones((8, 128)) * 0.5
    for i in range(8):
        x3 = x3.at[i, :10].set(0.9)   # Top 10 higher
        x3 = x3.at[i, 10:30].set(0.5)  # Next 20 same
        x3 = x3.at[i, 30:].set(0.1)    # Rest lower
    assert compare_with_jax_topk(x3, 15, "Extreme ties", verbose=False)
    print(f"  ✅ Pass")

    # Test 4: k crosses tie boundary
    print("\nTest 4: k crosses tie boundary")
    x4 = jnp.array([
        [1.0]*50 + [0.8]*30 + [0.5]*20 + [0.3]*20 + [0.1]*8
    ], dtype=jnp.float32)
    x4 = jnp.tile(x4, (4, 1))
    # k=70 means we need all 1.0s (50) and all 0.8s (30), stopping at 80
    # But we want exactly 70, so we get 50 + 20 of the 0.8s
    assert compare_with_jax_topk(x4, 70, "k crosses tie boundary", verbose=False)
    print(f"  ✅ Pass")

    # Test 5: All same values (extreme tie)
    print("\nTest 5: All same values")
    x5 = jnp.ones((4, 256)) * 0.5
    # With all same values, jax.lax.top_k will return the first k elements (stable sort)
    assert compare_with_jax_topk(x5, 50, "All same", verbose=False)
    print(f"  ✅ Pass")

    print("\n✅ All tie tests passed!\n")


def test_edge_cases():
    """Test edge cases."""
    print("="*70)
    print("Test Suite: Edge Cases")
    print("="*70)

    # Test 1: k = 1
    print("\nTest 1: k=1 (smallest possible)")
    x1 = jnp.array([[3.0, 1.0, 4.0, 1.0, 5.0, 9.0]])
    assert compare_with_jax_topk(x1, 1, "k=1", verbose=False)
    print("  ✅ Pass")

    # Test 2: k = vocab_size - 1
    print("\nTest 2: k close to vocab_size")
    x2 = jax.random.uniform(jax.random.PRNGKey(42), (4, 128))
    assert compare_with_jax_topk(x2, 127, "k=127 out of 128", verbose=False)
    print("  ✅ Pass")

    # Test 3: With negative values
    print("\nTest 3: Mixed positive and negative values")
    x3 = jax.random.uniform(jax.random.PRNGKey(42), (8, 256)) * 2 - 1  # Range [-1, 1]
    assert compare_with_jax_topk(x3, 50, "Negative values", verbose=False)
    print("  ✅ Pass")

    # Test 4: Very small values
    print("\nTest 4: Very small values")
    x4 = jax.random.uniform(jax.random.PRNGKey(42), (4, 128)) * 1e-6
    assert compare_with_jax_topk(x4, 20, "Small values", verbose=False)
    print("  ✅ Pass")

    # Test 5: With inf and -inf
    print("\nTest 5: With inf values")
    x5 = jax.random.uniform(jax.random.PRNGKey(42), (4, 128))
    x5 = x5.at[0, 0].set(jnp.inf)
    x5 = x5.at[1, 0].set(-jnp.inf)
    assert compare_with_jax_topk(x5, 20, "Inf values", verbose=False)
    print("  ✅ Pass")

    print("\n✅ All edge case tests passed!\n")


def run_stress_test():
    """Run stress tests with many random cases."""
    print("="*70)
    print("Stress Test: 100 random configurations")
    print("="*70)

    rng = jax.random.PRNGKey(999)
    failures = 0

    for i in range(100):
        rng, subkey = jax.random.split(rng)

        # Random configuration
        batch_size = jax.random.choice(subkey, jnp.array([4, 8, 16]), shape=()).item()
        rng, subkey = jax.random.split(rng)
        vocab_size = jax.random.choice(subkey, jnp.array([128, 256, 512, 1024, 2048]), shape=()).item()
        rng, subkey = jax.random.split(rng)
        k = jax.random.randint(subkey, shape=(), minval=1, maxval=min(100, vocab_size)).item()

        rng, subkey = jax.random.split(rng)
        x = jax.random.uniform(subkey, (batch_size, vocab_size))

        # Randomly add ties
        if i % 3 == 0:
            rng, subkey = jax.random.split(rng)
            granularity = jax.random.choice(subkey, jnp.array([5, 10, 20])).item()
            x = jnp.round(x * granularity) / granularity

        test_name = f"  Config {i+1}: batch={batch_size}, vocab={vocab_size}, k={k}"
        if not compare_with_jax_topk(x, k, test_name, verbose=False):
            failures += 1
            print(f"    ❌ Failed")
        elif (i + 1) % 10 == 0:
            print(f"  ✅ Completed {i+1}/100 tests")

    if failures == 0:
        print(f"\n✅ All 100 stress tests passed!\n")
    else:
        print(f"\n❌ {failures}/100 tests failed\n")

    return failures == 0


def run_all_comprehensive_tests():
    """Run all comprehensive tests."""
    print("\n" + "="*70)
    print("COMPREHENSIVE TOPK_MASK_PALLAS TEST SUITE")
    print("="*70 + "\n")

    try:
        test_larger_shapes()
        test_with_ties()
        test_edge_cases()
        stress_passed = run_stress_test()

        if stress_passed:
            print("="*70)
            print("✅ ALL COMPREHENSIVE TESTS PASSED!")
            print("="*70)
            return True
        else:
            print("="*70)
            print("❌ SOME TESTS FAILED")
            print("="*70)
            return False

    except Exception as e:
        print(f"\n❌ Test suite failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_comprehensive_tests()
    exit(0 if success else 1)
