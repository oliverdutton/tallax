"""Test TPU lowering on CPU to analyze compilation without running on hardware."""

import os
os.environ["JAX_PALLAS_CPU_LOWER_AS_TPU"] = "1"

import time
import jax
import jax.numpy as jnp
import numpy as np
from tallax.vllm import topk_topp_and_sample, top_p_and_sample
from tallax.vllm.tpu_inference_sampling_as_standalone_file import TPUSupportedSamplingMetadata
from tallax.tax import bitonic_top_k
from tallax.tax.divide_and_filter_topk.topk import top_bounded_k


def time_lowering_only(fn, *args, name=None, fn_kwargs=None):
    """Time only the lowering stage (without compilation or execution).

    Args:
        fn: The function to lower (should be already jitted)
        *args: Arguments to pass to the function
        name: Optional name for display
        fn_kwargs: Keyword arguments to pass to the function (not to .lower())
    """
    fn_kwargs = fn_kwargs or {}
    func_name = name or getattr(fn, '__name__', 'unnamed_function')

    print(f"\n{'='*60}")
    print(f"Timing lowering for: {func_name}")
    print(f"{'='*60}")

    # Time lowering
    t0 = time.perf_counter()
    try:
        # Call .lower() with lowering_platforms parameter
        if hasattr(fn, 'lower'):
            # Already jitted - use it directly
            lowered = fn.lower(*args, **fn_kwargs, lowering_platforms=("tpu",))
        else:
            # Need to jit first
            jitted = jax.jit(fn)
            lowered = jitted.lower(*args, **fn_kwargs, lowering_platforms=("tpu",))

        t1 = time.perf_counter()
        lower_time = t1 - t0
        print(f"  Lowering time: {lower_time*1000:.2f} ms")
        print(f"  ✓ SUCCESS")
        success = True

        # Try to get some info about the lowered code
        try:
            hlo_text = lowered.as_text()
            print(f"  HLO size: {len(hlo_text)} characters")
        except:
            pass

    except Exception as e:
        t1 = time.perf_counter()
        lower_time = t1 - t0
        print(f"  Lowering time: {lower_time*1000:.2f} ms")
        print(f"  ✗ FAILED")
        error_msg = str(e)
        if len(error_msg) > 200:
            error_msg = error_msg[:200] + "..."
        print(f"  Error: {type(e).__name__}: {error_msg}")
        success = False

    print(f"{'='*60}\n")

    return {
        'lower': lower_time,
        'success': success,
    }


def test_shape_lowering(shape, dtype=jnp.bfloat16, seed=42):
    """Test lowering timings for a specific shape."""
    num_tokens, vocab_size = shape

    print(f"\n{'#'*70}")
    print(f"Testing shape={shape}, dtype={dtype}")
    print(f"{'#'*70}")

    # Setup test data
    key = jax.random.PRNGKey(seed)
    key, topk_key, topp_key, temp_key, logits_key, sample_key = jax.random.split(key, 6)

    tpu_sampling_metadata = TPUSupportedSamplingMetadata(
        top_k=jax.random.randint(topk_key, (num_tokens,), 1, 128, dtype=jnp.int32),
        top_p=jax.random.uniform(topp_key, (num_tokens,), dtype=jnp.float32),
        temperature=10 ** jax.random.normal(temp_key, (num_tokens,), dtype=jnp.float32),
        do_sampling=True,
    )

    logits = jax.random.normal(logits_key, shape).astype(dtype)

    # Test 1: top_bounded_k (the main suspect)
    print("\n--- Test 1: top_bounded_k (divide-and-filter topk) ---")
    timings_bounded = time_lowering_only(
        top_bounded_k,
        logits,
        tpu_sampling_metadata.top_k,
        name="top_bounded_k",
        fn_kwargs={
            'max_k': 128,
            'num_bins': 256,
            'bins_topm_schedule': (5, 9),
            'guarantee_convergence': True,
            'replace_val': -1e12,
        }
    )

    # Test 2: bitonic_top_k (for comparison)
    print("\n--- Test 2: bitonic_top_k (reference) ---")
    timings_bitonic = time_lowering_only(
        bitonic_top_k,
        logits,
        name="bitonic_top_k",
        fn_kwargs={'k': 128}
    )

    # Test 3: Full pipeline
    print("\n--- Test 3: topk_topp_and_sample (full pipeline) ---")
    timings_full = time_lowering_only(
        topk_topp_and_sample,
        sample_key,
        logits,
        tpu_sampling_metadata,
        name="topk_topp_and_sample"
    )

    return {
        'bounded': timings_bounded,
        'bitonic': timings_bitonic,
        'full': timings_full,
    }


def main():
    print("=" * 70)
    print("TPU Lowering on CPU - Batch Size Comparison")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print(f"JAX_PALLAS_CPU_LOWER_AS_TPU: {os.environ.get('JAX_PALLAS_CPU_LOWER_AS_TPU', 'not set')}")

    # Test both shapes
    shapes = [(16, 2048), (256, 2048)]
    results = {}

    for shape in shapes:
        results[shape] = test_shape_lowering(shape)

    # Summary comparison
    print("\n" + "=" * 70)
    print("LOWERING TIME COMPARISON")
    print("=" * 70)

    for component in ['bounded', 'bitonic', 'full']:
        print(f"\n{component.upper()}:")

        times = []
        for shape in shapes:
            if results[shape][component]['success']:
                lower_ms = results[shape][component]['lower'] * 1000
                print(f"  {shape}: {lower_ms:.0f} ms")
                times.append(results[shape][component]['lower'])
            else:
                print(f"  {shape}: FAILED")

        # Calculate ratio if both succeeded
        if len(times) == 2:
            ratio = times[1] / times[0]
            print(f"  → Ratio (256,2048)/(16,2048): {ratio:.2f}x")

    # Additional analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Calculate grid sizes
    block_token = 8
    for shape in shapes:
        num_tokens = shape[0]
        num_programs = (num_tokens + block_token - 1) // block_token
        print(f"\n{shape}:")
        print(f"  Grid size (num_programs): {num_programs}")
        print(f"  block_token: {block_token}")


if __name__ == "__main__":
    main()
