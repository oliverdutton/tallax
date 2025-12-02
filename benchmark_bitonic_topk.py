#!/usr/bin/env python3
"""
Benchmark and validation script for bitonic_topk.

Tests the Pallas-based bitonic_topk implementation against JAX's XLA top_k
for correctness and performance on various input shapes with int32 values.
"""

import time
import jax
import jax.numpy as jnp
from tallax.tax.bitonic_topk import bitonic_topk

# Test configurations
NUM_TOKENS = [8, 16, 32, 64, 128]
VOCAB_SIZES = [128, 256, 512, 1024, 2048, 4096, 8192]

NUM_WARMUP = 3
NUM_ITERATIONS = 10


def validate_results(pallas_values, xla_values, shape, test_name):
    """Validate that Pallas result matches XLA reference."""
    # Check shapes match
    if pallas_values.shape != xla_values.shape:
        print(f"  ❌ Shape mismatch! Pallas: {pallas_values.shape}, XLA: {xla_values.shape}")
        return False

    # Check values match
    matches = jnp.allclose(pallas_values, xla_values)

    if matches:
        print(f"  ✅ Values match exactly")
        return True
    else:
        # Check if it's just ties (all values are correct but ordering differs)
        pallas_sorted = jnp.sort(pallas_values, axis=-1)[:, ::-1]
        xla_sorted = jnp.sort(xla_values, axis=-1)[:, ::-1]

        if jnp.allclose(pallas_sorted, xla_sorted):
            print(f"  ⚠️  Values match but ordering differs (ties)")
            return True
        else:
            print(f"  ❌ FAILED - Values don't match!")
            print(f"     First row Pallas: {pallas_values[0, :10]}")
            print(f"     First row XLA:    {xla_values[0, :10]}")

            # Check for zeros (common bug)
            if jnp.all(pallas_values == 0):
                print(f"     ⚠️  Pallas returned all zeros!")

            return False


def benchmark_function(func, *args, name="", warmup=NUM_WARMUP, iterations=NUM_ITERATIONS):
    """Benchmark a JAX function."""
    # Warmup
    for _ in range(warmup):
        result = func(*args)
        if isinstance(result, tuple):
            result[0].block_until_ready()
        else:
            result.block_until_ready()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func(*args)
        if isinstance(result, tuple):
            result[0].block_until_ready()
        else:
            result.block_until_ready()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    mean_time = jnp.mean(jnp.array(times))
    std_time = jnp.std(jnp.array(times))

    if name:
        print(f"  {name}: {mean_time:.3f}ms ± {std_time:.3f}ms")

    return result, mean_time, std_time


def run_tests():
    """Run all validation and benchmark tests."""
    print("=" * 80)
    print("BITONIC TOP-K VALIDATION & BENCHMARK")
    print("=" * 80)
    print(f"Device: {jax.default_backend()}")
    print(f"Token counts: {NUM_TOKENS}")
    print(f"Vocab sizes: {VOCAB_SIZES}")
    print("=" * 80)

    # Special test: iota sort
    print("\n--- IOTA Test (8, 128) ---")
    iota_data = jax.lax.broadcasted_iota(jnp.int32, (8, 128), 1)
    print(f"Input: iota pattern (0-127 repeated for 8 rows)")
    pallas_iota = bitonic_topk(iota_data, k=128, descending=True)
    xla_iota = jax.vmap(lambda y: jax.lax.top_k(y, 128))(iota_data)
    print(f"Pallas result shape: {pallas_iota[0].shape}")
    print(f"XLA result shape: {xla_iota[0].shape}")
    validate_results(pallas_iota[0], xla_iota[0], (8, 128), "iota")

    all_passed = True
    results = []

    for tokens in NUM_TOKENS:
        # Skip if tokens > 128 (limitation)
        if tokens > 128:
            print(f"\nSkipping tokens={tokens} (exceeds NUM_LANES=128 limit)")
            continue

        for vocab in VOCAB_SIZES:
            shape = (tokens, vocab)
            print(f"\n{'='*80}")
            print(f"Testing shape: {shape}")
            print(f"{'='*80}")

            # Generate test data - using negative random permutation like user
            total_size = tokens * vocab
            key = jax.random.PRNGKey(42)
            x = -jax.random.permutation(key, total_size).reshape(shape).astype(jnp.int32)
            index = jax.lax.broadcasted_iota(jnp.int32, shape, 1)

            # Test 1: Values only
            print(f"\n[1] Values only (int32, descending)")
            try:
                pallas_result, pallas_time, _ = benchmark_function(
                    lambda data: bitonic_topk(data, k=128, descending=True),
                    x,
                    name="Pallas"
                )

                xla_result, xla_time, _ = benchmark_function(
                    lambda data: jax.vmap(lambda y: jax.lax.top_k(y, 128))(data),
                    x,
                    name="XLA   "
                )

                # Extract values
                pallas_values = pallas_result[0] if isinstance(pallas_result, tuple) else pallas_result
                xla_values = xla_result[0]

                passed = validate_results(pallas_values, xla_values, shape, "values-only")
                all_passed = all_passed and passed

                speedup = xla_time / pallas_time
                print(f"  Speedup: {speedup:.2f}x {'🚀' if speedup > 1 else '🐢'}")

                results.append({
                    'tokens': tokens,
                    'vocab': vocab,
                    'test': 'values',
                    'passed': passed,
                    'pallas_ms': pallas_time,
                    'xla_ms': xla_time,
                    'speedup': speedup
                })
            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                all_passed = False

            # Test 2: With indices
            print(f"\n[2] With indices (int32, descending)")
            try:
                pallas_result, pallas_time, _ = benchmark_function(
                    lambda data, idx: bitonic_topk((data, idx), k=128, num_keys=1, descending=True),
                    x, index,
                    name="Pallas"
                )

                def xla_topk_with_indices(data, idx):
                    def per_row(values, inds):
                        sorted_vals, sorted_idx = jax.lax.top_k(values, 128)
                        return sorted_vals, jnp.take(inds, sorted_idx)
                    return jax.vmap(per_row)(data, idx)

                xla_result, xla_time, _ = benchmark_function(
                    xla_topk_with_indices,
                    x, index,
                    name="XLA   "
                )

                # Validate values
                pallas_values, pallas_indices = pallas_result
                xla_values, xla_indices = xla_result

                passed_values = validate_results(pallas_values, xla_values, shape, "with-indices (values)")

                # Validate indices are in valid range
                indices_valid = jnp.all((pallas_indices >= 0) & (pallas_indices < vocab))
                if indices_valid:
                    print(f"  ✅ Indices in valid range [0, {vocab})")
                else:
                    print(f"  ❌ Indices out of range!")
                    print(f"     Min: {jnp.min(pallas_indices)}, Max: {jnp.max(pallas_indices)}")
                    passed_values = False

                all_passed = all_passed and passed_values and indices_valid

                speedup = xla_time / pallas_time
                print(f"  Speedup: {speedup:.2f}x {'🚀' if speedup > 1 else '🐢'}")

                results.append({
                    'tokens': tokens,
                    'vocab': vocab,
                    'test': 'indices',
                    'passed': passed_values and indices_valid,
                    'pallas_ms': pallas_time,
                    'xla_ms': xla_time,
                    'speedup': speedup
                })
            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                all_passed = False

    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Tokens':<8} {'Vocab':<8} {'Test':<10} {'Status':<10} {'Pallas':<12} {'XLA':<12} {'Speedup':<10}")
    print("-" * 80)

    for r in results:
        status = "✅ PASS" if r['passed'] else "❌ FAIL"
        print(f"{r['tokens']:<8} {r['vocab']:<8} {r['test']:<10} {status:<10} "
              f"{r['pallas_ms']:>8.2f}ms {r['xla_ms']:>8.2f}ms {r['speedup']:>8.2f}x")

    print("=" * 80)

    # Statistics
    if results:
        passed_count = sum(1 for r in results if r['passed'])
        total_count = len(results)
        pass_rate = (passed_count / total_count) * 100

        avg_speedup = jnp.mean(jnp.array([r['speedup'] for r in results]))

        print(f"Pass rate: {passed_count}/{total_count} ({pass_rate:.1f}%)")
        print(f"Average speedup: {avg_speedup:.2f}x")

    print("=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
