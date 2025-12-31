"""Test optimization barriers with interpret=True on CPU."""

import time
import jax
import jax.numpy as jnp
from tallax.vllm.tpu_inference_sampling_as_standalone_file import TPUSupportedSamplingMetadata
from tallax.tax.divide_and_filter_topk.topk import top_bounded_k


def test_with_interpret(shape, bins_topm_unroll=64, use_barriers=False, dtype=jnp.bfloat16, seed=42):
    """Test with interpret=True to run on CPU."""
    num_tokens, vocab_size = shape

    print(f"\n{'='*70}")
    print(f"Shape: {shape}, unroll: {bins_topm_unroll}, barriers: {use_barriers}")
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

    # Time the call
    print("Running with interpret=True (CPU)...")
    t0 = time.perf_counter()

    result = top_bounded_k(
        logits,
        k,
        max_k=128,
        block_token=8,
        num_bins=256,
        bins_topm_schedule=(5, 9),
        bins_topm_unroll=bins_topm_unroll,
        guarantee_convergence=True,
        replace_val=-1e12,
        interpret=True,  # <-- Key change for CPU
    )

    t1 = time.perf_counter()

    total_time = t1 - t0
    print(f"Total time: {total_time:.2f}s")
    print(f"Result shape: {result[0].shape}")

    return total_time


def main():
    print("="*70)
    print("Testing with interpret=True on CPU")
    print("="*70)
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")

    # Test 1: Different unroll values on small shape
    print("\n" + "="*70)
    print("TEST 1: Different unroll values (16, 2048)")
    print("="*70)

    unroll_values = [16, 32, 64]
    results_unroll = {}

    for unroll in unroll_values:
        results_unroll[unroll] = test_with_interpret(
            shape=(16, 2048),
            bins_topm_unroll=unroll,
            use_barriers=False
        )

    # Summary
    print("\n" + "="*70)
    print("RESULTS - Unroll Impact")
    print("="*70)
    print(f"{'Unroll':<10} {'Time (s)':<15} {'Speedup vs 64':<15}")
    print("-"*40)

    baseline = results_unroll.get(64)
    for unroll in unroll_values:
        time_val = results_unroll[unroll]
        if baseline:
            speedup = baseline / time_val
            print(f"{unroll:<10} {time_val:<15.2f} {speedup:<15.2f}x")

    print("\nNote: interpret=True is VERY slow and doesn't reflect compilation.")
    print("These timings are for correctness checking only.")
    print("Run on TPU without interpret=True for real performance tests.")


if __name__ == "__main__":
    main()
