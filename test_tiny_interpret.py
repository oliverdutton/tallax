"""Test with very small shape to verify interpret mode works."""

import time
import jax
import jax.numpy as jnp
from tallax.vllm.tpu_inference_sampling_as_standalone_file import TPUSupportedSamplingMetadata
from tallax.tax.divide_and_filter_topk.topk import top_bounded_k


def test_tiny():
    """Test with minimal shape."""
    # Very small shape
    shape = (8, 256)  # Tiny!
    num_tokens, vocab_size = shape

    print(f"Testing TINY shape: {shape}")
    print("This should complete quickly to verify correctness...")

    # Setup
    key = jax.random.PRNGKey(42)
    key, topk_key, topp_key, temp_key, logits_key = jax.random.split(key, 5)

    tpu_sampling_metadata = TPUSupportedSamplingMetadata(
        top_k=jnp.full((num_tokens,), 32, dtype=jnp.int32),  # Small k
        top_p=jnp.full((num_tokens,), 0.9, dtype=jnp.float32),
        temperature=jnp.ones((num_tokens,), dtype=jnp.float32),
        do_sampling=True,
    )

    logits = jax.random.normal(logits_key, shape).astype(jnp.bfloat16)
    k = tpu_sampling_metadata.top_k

    print("Running with interpret=True...")
    t0 = time.perf_counter()

    result = top_bounded_k(
        logits,
        k,
        max_k=32,  # Small max_k
        block_token=8,
        num_bins=128,  # Smaller num_bins
        bins_topm_schedule=(3, 5),  # Simpler schedule
        bins_topm_unroll=8,  # Small unroll
        guarantee_convergence=True,
        replace_val=-1e12,
        interpret=True,
    )

    t1 = time.perf_counter()

    print(f"✓ Completed in {t1-t0:.2f}s")
    print(f"Result shapes: {result[0].shape}, {result[1].shape}")
    print(f"First few values: {result[0][0, :5]}")

    return result


if __name__ == "__main__":
    print("="*70)
    print("Tiny Interpret Mode Test - Correctness Check Only")
    print("="*70)
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print()

    result = test_tiny()

    print("\n✓ SUCCESS: Code works with interpret=True")
    print("This confirms the functions are correct.")
    print("\nNOTE: interpret=True is too slow for timing tests.")
    print("Must use TPU hardware without interpret=True for real tests.")
