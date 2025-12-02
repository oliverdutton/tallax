#!/usr/bin/env python3
"""
Quick benchmark and validation script for bitonic_top_k.

For comprehensive testing, use: pytest tests/bitonic_top_k_test.py -v
For benchmarking, run this script directly.
"""

import jax
import jax.numpy as jnp
from tallax.tax.bitonic_top_k import bitonic_top_k
from tallax.utils import is_cpu_platform
from tallax.test_utils import benchmark, verify_topk_output


def quick_validation():
    """Quick validation across key shapes."""
    print("="*80)
    print("BITONIC TOP-K QUICK VALIDATION")
    print("="*80)

    test_shapes = [
        (8, 128),
        (8, 256),
        (8, 1024),
        (8, 2048),
        (16, 2048),
        (64, 2048),
    ]

    all_passed = True

    for shape in test_shapes:
        num_tokens, vocab_size = shape
        print(f"\nTesting {shape}...")

        # Generate test data
        key = jax.random.PRNGKey(42)
        total_size = num_tokens * vocab_size
        x = -jax.random.permutation(key, total_size).reshape(shape).astype(jnp.int32)

        # Test values only
        try:
            interpret = is_cpu_platform()
            result = bitonic_top_k(x, k=128, descending=True, interpret=interpret)
            xla_result = jax.vmap(lambda y: jax.lax.top_k(y, 128))(x)

            pallas_values = result[0] if isinstance(result, tuple) else result
            xla_values = xla_result[0]

            # Check sorted values match
            pallas_sorted = jnp.sort(pallas_values, axis=-1)[:, ::-1]
            xla_sorted = jnp.sort(xla_values, axis=-1)[:, ::-1]

            if jnp.allclose(pallas_sorted, xla_sorted):
                print(f"  ✅ Values-only: PASSED")
            else:
                print(f"  ❌ Values-only: FAILED")
                all_passed = False
        except Exception as e:
            print(f"  ❌ Values-only: ERROR - {e}")
            all_passed = False

        # Test with indices
        try:
            indices = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
            interpret = is_cpu_platform()
            result = bitonic_top_k((x, indices), k=128, num_keys=1, descending=True, interpret=interpret)
            validation = verify_topk_output(x, result)

            if bool(validation.all()):
                print(f"  ✅ With-indices: PASSED")
            else:
                print(f"  ❌ With-indices: FAILED ({int(validation.sum())}/{num_tokens} rows)")
                all_passed = False
        except Exception as e:
            print(f"  ❌ With-indices: ERROR - {e}")
            all_passed = False

    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED")
    else:
        print("⚠️  SOME VALIDATIONS FAILED")
    print("="*80)

    return all_passed


def quick_benchmark():
    """Quick benchmark of key shapes."""
    print("\n" + "="*80)
    print("BITONIC TOP-K QUICK BENCHMARK")
    print("="*80)

    benchmark_shapes = [
        (8, 2048),
        (16, 2048),
        (32, 2048),
        (64, 2048),
    ]

    for shape in benchmark_shapes:
        num_tokens, vocab_size = shape
        print(f"\n{shape}:")

        # Generate test data
        key = jax.random.PRNGKey(42)
        total_size = num_tokens * vocab_size
        x = -jax.random.permutation(key, total_size).reshape(shape).astype(jnp.int32)

        print("  Pallas bitonic_top_k:")
        interpret = is_cpu_platform()
        benchmark(lambda: bitonic_top_k(x, k=128, descending=True, interpret=interpret))

        print("  XLA top_k:")
        benchmark(lambda: jax.vmap(lambda y: jax.lax.top_k(y, 128))(x))


if __name__ == "__main__":
    success = quick_validation()
    quick_benchmark()
    exit(0 if success else 1)

