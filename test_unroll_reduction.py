"""Test if reducing bins_topm_unroll reduces compilation time.

The hypothesis is that loop unrolling creates more jaxpr equations,
which slows down the jaxpr creation and lowering stages.
"""

import time
import jax
import jax.numpy as jnp
from tallax.vllm.tpu_inference_sampling_as_standalone_file import TPUSupportedSamplingMetadata
from tallax.tax.divide_and_filter_topk.topk import top_bounded_k


def test_unroll_values(shape, unroll_values, dtype=jnp.bfloat16, seed=42):
    """Test different bins_topm_unroll values."""
    num_tokens, vocab_size = shape

    print(f"\n{'='*70}")
    print(f"Testing shape={shape} with different unroll values")
    print(f"{'='*70}")

    # Setup
    key = jax.random.PRNGKey(seed)
    key, topk_key, topp_key, temp_key, logits_key = jax.random.split(key, 5)

    tpu_sampling_metadata = TPUSupportedSamplingMetadata(
        top_k=jax.random.randint(topk_key, (num_tokens,), 1, 128, dtype=jnp.int32),
        top_p=jax.random.uniform(topp_key, (num_tokens,), dtype=jnp.float32),
        temperature=10 ** jax.random.normal(temp_key, (num_tokens,), dtype=jnp.float32),
        do_sampling=True,
    )

    logits = jax.random.normal(logits_key, shape).astype(dtype)
    k = tpu_sampling_metadata.top_k

    results = {}

    for unroll in unroll_values:
        print(f"\n--- Testing bins_topm_unroll={unroll} ---")

        # Clear any JAX caches
        jax.clear_caches()

        t0 = time.perf_counter()
        try:
            result = top_bounded_k(
                logits,
                k,
                max_k=128,
                block_token=8,
                num_bins=256,
                bins_topm_schedule=(5, 9),
                bins_topm_unroll=unroll,  # <-- Variable
                guarantee_convergence=True,
                replace_val=-1e12,
                interpret=False,
            )
            t1 = time.perf_counter()

            compile_time = t1 - t0
            print(f"  Compilation time: {compile_time:.2f}s")
            results[unroll] = compile_time

        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")
            results[unroll] = None

    return results


def main():
    print("="*70)
    print("Testing bins_topm_unroll Impact on Compilation Time")
    print("="*70)
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")

    # Test different unroll values
    unroll_values = [8, 16, 32, 64]
    shapes = [(16, 2048), (256, 2048)]

    all_results = {}

    for shape in shapes:
        all_results[shape] = test_unroll_values(shape, unroll_values)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Compilation Time by Unroll Value")
    print("="*70)

    for shape in shapes:
        print(f"\nShape {shape}:")
        print(f"{'unroll':<10} {'time (s)':<15} {'speedup vs 64':<15}")
        print("-"*40)

        baseline = all_results[shape].get(64)
        for unroll in unroll_values:
            time_val = all_results[shape].get(unroll)
            if time_val is not None:
                if baseline is not None and baseline > 0:
                    speedup = baseline / time_val
                    print(f"{unroll:<10} {time_val:<15.2f} {speedup:<15.2f}x")
                else:
                    print(f"{unroll:<10} {time_val:<15.2f} {'N/A':<15}")
            else:
                print(f"{unroll:<10} {'FAILED':<15} {'N/A':<15}")

    print("\n" + "="*70)
    print("EXPECTED RESULTS:")
    print("  - Smaller unroll → Faster jaxpr creation (less Python overhead)")
    print("  - Smaller unroll → Possibly slower runtime (less loop unrolling)")
    print("  - Tradeoff between compilation time and runtime performance")
    print("="*70)


if __name__ == "__main__":
    main()
