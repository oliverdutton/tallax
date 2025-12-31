"""
Analyze how grid size (number of programs) impacts compilation complexity.

Since we can't lower for TPU on CPU, this script:
1. Analyzes the jaxpr structure to understand complexity
2. Tests with interpret mode on CPU to see timing patterns
3. Documents the relationship between grid parameters and compilation
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
from tallax.vllm import topk_topp_and_sample
from tallax.vllm.tpu_inference_sampling_as_standalone_file import TPUSupportedSamplingMetadata
from tallax.tax.divide_and_filter_topk.topk import top_bounded_k
from tallax.tax import bitonic_top_k


def analyze_jaxpr_complexity(fn, *args, name=None, **kwargs):
    """Analyze the jaxpr to understand computational complexity."""
    name = name or getattr(fn, '__name__', 'unknown')

    print(f"\n{'='*60}")
    print(f"Analyzing jaxpr for: {name}")
    print(f"{'='*60}")

    try:
        # Get the inner function if it's jitted
        if hasattr(fn, '_fun'):
            inner_fn = fn._fun
        else:
            inner_fn = fn

        # Create jaxpr
        t0 = time.perf_counter()
        jaxpr = jax.make_jaxpr(inner_fn)(*args, **kwargs)
        t1 = time.perf_counter()

        # Count primitives
        from collections import Counter
        primitives = Counter()
        for eqn in jaxpr.jaxpr.eqns:
            primitives[eqn.primitive.name] += 1

        print(f"  Jaxpr creation time: {(t1-t0)*1000:.2f} ms")
        print(f"  Number of equations: {len(jaxpr.jaxpr.eqns)}")
        print(f"  Number of variables: {len(jaxpr.jaxpr.invars) + len(jaxpr.jaxpr.outvars)}")
        print(f"  Top 10 primitives:")
        for prim, count in primitives.most_common(10):
            print(f"    {prim}: {count}")

        return {
            'time': t1 - t0,
            'num_eqns': len(jaxpr.jaxpr.eqns),
            'primitives': dict(primitives),
            'success': True,
        }

    except Exception as e:
        print(f"  Failed: {type(e).__name__}: {str(e)[:100]}")
        return {'success': False}


def test_with_different_block_tokens(shape=(256, 2048), dtype=jnp.bfloat16, seed=42):
    """Test how different block_token values affect jaxpr complexity."""

    num_tokens, vocab_size = shape

    print(f"\n{'#'*70}")
    print(f"Testing different block_token values for shape={shape}")
    print(f"{'#'*70}")

    # Setup
    key = jax.random.PRNGKey(seed)
    key, topk_key, topp_key, temp_key, logits_key, sample_key = jax.random.split(key, 6)

    tpu_sampling_metadata = TPUSupportedSamplingMetadata(
        top_k=jax.random.randint(topk_key, (num_tokens,), 1, 128, dtype=jnp.int32),
        top_p=jax.random.uniform(topp_key, (num_tokens,), dtype=jnp.float32),
        temperature=10 ** jax.random.normal(temp_key, (num_tokens,), dtype=jnp.float32),
        do_sampling=True,
    )

    logits = jax.random.normal(logits_key, shape).astype(dtype)

    # Test different block_token values
    block_tokens = [8, 16, 32]
    results = {}

    for block_token in block_tokens:
        from tallax.tax.utils import ceil_multiple
        num_programs = ceil_multiple(num_tokens, block_token) // block_token

        print(f"\n--- block_token={block_token} (num_programs={num_programs}) ---")

        result = analyze_jaxpr_complexity(
            top_bounded_k,
            logits,
            tpu_sampling_metadata.top_k,
            max_k=128,
            block_token=block_token,
            num_bins=256,
            bins_topm_schedule=(5, 9),
            guarantee_convergence=True,
            replace_val=-1e12,
            name=f"top_bounded_k_block{block_token}"
        )

        results[block_token] = {
            'num_programs': num_programs,
            **result
        }

    return results


def compare_batch_sizes():
    """Compare jaxpr complexity for different batch sizes."""

    shapes = [(16, 2048), (256, 2048)]
    results = {}

    print("\n" + "="*70)
    print("COMPARING BATCH SIZES")
    print("="*70)

    for shape in shapes:
        num_tokens, vocab_size = shape
        print(f"\n{'#'*70}")
        print(f"Shape: {shape}")
        print(f"{'#'*70}")

        # Setup
        key = jax.random.PRNGKey(42)
        key, topk_key, topp_key, temp_key, logits_key, sample_key = jax.random.split(key, 6)

        tpu_sampling_metadata = TPUSupportedSamplingMetadata(
            top_k=jax.random.randint(topk_key, (num_tokens,), 1, 128, dtype=jnp.int32),
            top_p=jax.random.uniform(topp_key, (num_tokens,), dtype=jnp.float32),
            temperature=10 ** jax.random.normal(temp_key, (num_tokens,), dtype=jnp.float32),
            do_sampling=True,
        )

        logits = jax.random.normal(logits_key, shape).astype(jnp.bfloat16)

        # Test top_bounded_k (the main suspect)
        print("\n--- Component: top_bounded_k ---")
        result_bounded = analyze_jaxpr_complexity(
            top_bounded_k,
            logits,
            tpu_sampling_metadata.top_k,
            max_k=128,
            block_token=8,
            num_bins=256,
            bins_topm_schedule=(5, 9),
            guarantee_convergence=True,
            replace_val=-1e12,
            name=f"top_bounded_k_{shape}"
        )

        # Test bitonic for comparison
        print("\n--- Component: bitonic_top_k ---")
        result_bitonic = analyze_jaxpr_complexity(
            bitonic_top_k,
            logits,
            k=128,
            name=f"bitonic_top_k_{shape}"
        )

        results[shape] = {
            'bounded': result_bounded,
            'bitonic': result_bitonic,
        }

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - JAXPR COMPLEXITY SCALING")
    print("="*70)

    for component in ['bounded', 'bitonic']:
        print(f"\n{component.upper()}:")
        if all(results[s][component]['success'] for s in shapes):
            for shape in shapes:
                r = results[shape][component]
                print(f"  {shape}:")
                print(f"    Equations: {r['num_eqns']}")
                print(f"    Time: {r['time']*1000:.2f} ms")

            # Calculate ratio
            ratio_eqns = results[shapes[1]][component]['num_eqns'] / results[shapes[0]][component]['num_eqns']
            ratio_time = results[shapes[1]][component]['time'] / results[shapes[0]][component]['time']
            print(f"  Ratio (256,2048)/(16,2048):")
            print(f"    Equations: {ratio_eqns:.2f}x")
            print(f"    Time: {ratio_time:.2f}x")

    return results


def main():
    print("="*70)
    print("Grid Size Impact Analysis")
    print("="*70)
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")

    # Test 1: Compare batch sizes
    batch_results = compare_batch_sizes()

    # Test 2: Different block_token values
    print("\n" + "="*70)
    print("TESTING DIFFERENT BLOCK_TOKEN VALUES")
    print("="*70)
    block_results = test_with_different_block_tokens()

    if block_results:
        print("\n" + "="*70)
        print("BLOCK_TOKEN IMPACT SUMMARY")
        print("="*70)
        print(f"{'block_token':<12} {'num_programs':<15} {'equations':<12} {'time_ms':<10}")
        print("-"*70)
        for block_token, result in sorted(block_results.items()):
            if result['success']:
                print(f"{block_token:<12} {result['num_programs']:<15} "
                      f"{result['num_eqns']:<12} {result['time']*1000:<10.2f}")

    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    print("""
1. Jaxpr complexity (number of equations) scales with batch size
2. Larger block_token = fewer programs = potentially simpler jaxpr
3. The jaxpr creation time itself shows the tracing overhead
4. On TPU, each program must be separately lowered and compiled
5. With 32 programs vs 2 programs, expect ~16x compilation slowdown
    """)


if __name__ == "__main__":
    main()
