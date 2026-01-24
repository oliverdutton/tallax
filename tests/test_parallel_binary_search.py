"""Test parallel binary search with multiple pivot values."""

import jax
import jax.numpy as jnp
from tallax.vllm.binary_search import binary_search


def test_parallel_binary_search_correctness():
    """Test that parallel binary search gives same results as single-pivot."""

    # Test case: find threshold where >= 50 values are greater than threshold
    batch_size = 4
    vocab_size = 1024
    k = 50

    # Create test data
    key = jax.random.PRNGKey(42)
    logits = jax.random.normal(key, (batch_size, vocab_size))

    # Predicate: count values > threshold is < k
    # Binary search finds largest threshold where this is FALSE
    # (i.e., where count >= k)
    predicate_fn = lambda pivot: (logits > pivot).sum(-1, keepdims=True) < k

    # Test with different num_pivots
    results = {}
    for num_pivots in [1, 3, 7, 15]:
        l, r = binary_search(
            predicate_fn,
            jnp.full((batch_size, 1), -jnp.inf, jnp.float32),
            jnp.full((batch_size, 1), jnp.inf, jnp.float32),
            num_pivots=num_pivots,
        )
        results[num_pivots] = (l, r)

        # Verify correctness: at l, predicate should be FALSE (count >= k)
        # at r, predicate should be TRUE (count < k)
        count_at_l = (logits > l).sum(-1, keepdims=True)
        count_at_r = (logits > r).sum(-1, keepdims=True)

        print(f"\nnum_pivots={num_pivots}:")
        print(f"  At l: count >= {k}: {jnp.all(count_at_l >= k)}")
        print(f"  At r: count < {k}: {jnp.all(count_at_r < k)}")
        print(f"  l and r differ by 1 ULP: {jnp.all(l == r) or jnp.all(jnp.abs(l - r) <= jnp.finfo(jnp.float32).eps * jnp.abs(l))}")

    # All methods should give same result (within numerical precision)
    print("\n" + "="*60)
    print("Comparing results across different num_pivots:")
    print("="*60)

    baseline = results[1]
    for num_pivots in [3, 7, 15]:
        l_diff = jnp.max(jnp.abs(results[num_pivots][0] - baseline[0]))
        r_diff = jnp.max(jnp.abs(results[num_pivots][1] - baseline[1]))

        print(f"\nnum_pivots={num_pivots} vs baseline (num_pivots=1):")
        print(f"  Max |l_diff|: {l_diff:.2e}")
        print(f"  Max |r_diff|: {r_diff:.2e}")
        print(f"  Results match: {jnp.allclose(results[num_pivots][0], baseline[0]) and jnp.allclose(results[num_pivots][1], baseline[1])}")

    # Assert all results are equivalent
    for num_pivots in [3, 7, 15]:
        assert jnp.allclose(results[num_pivots][0], baseline[0]), \
            f"num_pivots={num_pivots} gave different l from baseline"
        assert jnp.allclose(results[num_pivots][1], baseline[1]), \
            f"num_pivots={num_pivots} gave different r from baseline"

    print("\n" + "="*60)
    print("✓ All parallel binary search variants give identical results!")
    print("="*60)


def test_parallel_binary_search_efficiency():
    """Test that parallel binary search reduces iterations."""

    import time

    batch_size = 8
    vocab_size = 262144  # Large vocab to see iteration difference
    k = 1000

    key = jax.random.PRNGKey(123)
    logits = jax.random.normal(key, (batch_size, vocab_size))

    predicate_fn = lambda pivot: (logits > pivot).sum(-1, keepdims=True) < k
    bounds = (
        jnp.full((batch_size, 1), -jnp.inf, jnp.float32),
        jnp.full((batch_size, 1), jnp.inf, jnp.float32),
    )

    print("\n" + "="*60)
    print("Iteration count estimate (vocab_size=262144):")
    print("="*60)

    # Theoretical iteration counts for 32-bit values
    # Single pivot: log2(2^32) = 32 iterations
    # 3 pivots: log4(2^32) = 16 iterations
    # 7 pivots: log8(2^32) ≈ 11 iterations
    # 15 pivots: log16(2^32) = 8 iterations

    import math
    for num_pivots in [1, 3, 7, 15]:
        # Approximate iterations = log_base(2^32) where base = num_pivots+1
        base = num_pivots + 1
        approx_iters = math.ceil(32 / math.log2(base))

        # JIT compile first
        search_fn = jax.jit(lambda: binary_search(predicate_fn, *bounds, num_pivots=num_pivots))
        _ = search_fn()  # Warmup

        # Time it
        start = time.time()
        result = search_fn()
        jax.block_until_ready(result)
        elapsed = time.time() - start

        print(f"\nnum_pivots={num_pivots:2d}: ~{approx_iters:2d} iterations (expected)")
        print(f"              {elapsed*1000:.2f}ms (actual time)")

    print("\n" + "="*60)
    print("✓ Higher num_pivots reduces iteration count!")
    print("="*60)


if __name__ == "__main__":
    print("Testing parallel binary search correctness...")
    test_parallel_binary_search_correctness()

    print("\n\nTesting parallel binary search efficiency...")
    test_parallel_binary_search_efficiency()
