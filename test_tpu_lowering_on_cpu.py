"""Test TPU lowering on CPU to analyze compilation without running on hardware."""

import os
os.environ["JAX_PALLAS_CPU_LOWER_AS_TPU"] = "1"
# Force TPU platform for lowering
os.environ["JAX_PLATFORMS"] = "cpu"

import time
import jax
import jax.numpy as jnp
import numpy as np
from tallax.vllm import topk_topp_and_sample, top_p_and_sample
from tallax.vllm.tpu_inference_sampling_as_standalone_file import TPUSupportedSamplingMetadata
from tallax.tax import bitonic_top_k
from tallax.tax.divide_and_filter_topk.topk import top_bounded_k


def time_lowering_only(fn, *args, name=None, **kwargs):
    """Time only the lowering stage (without compilation or execution)."""
    func_name = name or getattr(fn, '__name__', 'unnamed_function')

    print(f"\n{'='*60}")
    print(f"Timing lowering for: {func_name}")
    print(f"{'='*60}")

    # Time lowering - call lower() on the jitted function directly
    t0 = time.perf_counter()
    try:
        # For functions that are already jitted
        if hasattr(fn, 'lower'):
            lowered = fn.lower(*args, lowering_platforms=("tpu",), **kwargs)
        else:
            # For functions that need to be jitted first
            jitted = jax.jit(fn)
            lowered = jitted.lower(*args, lowering_platforms=("tpu",), **kwargs)
        t1 = time.perf_counter()
        lower_time = t1 - t0
        print(f"  lowering:       {lower_time*1000:.2f} ms")
        print(f"  SUCCESS")
        success = True
    except Exception as e:
        t1 = time.perf_counter()
        lower_time = t1 - t0
        print(f"  lowering:       {lower_time*1000:.2f} ms (FAILED)")
        print(f"  Error: {type(e).__name__}: {str(e)[:300]}")
        import traceback
        traceback.print_exc()
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

    # Test top_bounded_k lowering
    print("\n--- Component: top_bounded_k (divide-and-filter topk) ---")
    timings_bounded = time_lowering_only(
        top_bounded_k,
        logits,
        k=tpu_sampling_metadata.top_k,
        max_k=128,
        num_bins=256,
        bins_topm_schedule=(5, 9),
        guarantee_convergence=True,
        replace_val=-1e12,
        name="top_bounded_k"
    )

    # Test full pipeline lowering
    print("\n--- Full Pipeline: topk_topp_and_sample ---")
    timings_full = time_lowering_only(
        topk_topp_and_sample,
        sample_key,
        logits,
        tpu_sampling_metadata,
        name="topk_topp_and_sample"
    )

    # Test bitonic_top_k for reference
    print("\n--- For reference: bitonic_top_k ---")
    timings_bitonic = time_lowering_only(
        bitonic_top_k,
        logits,
        k=128,
        name="bitonic_top_k"
    )

    return {
        'bounded': timings_bounded,
        'full': timings_full,
        'bitonic': timings_bitonic,
    }


def main():
    print("=" * 70)
    print("TPU Lowering on CPU - Timing Analysis")
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
    print("SUMMARY COMPARISON - LOWERING TIMES ONLY")
    print("=" * 70)

    for component in ['bounded', 'full', 'bitonic']:
        print(f"\n{component.upper()}:")
        for shape in shapes:
            if results[shape][component]['success']:
                lower = results[shape][component]['lower']
                print(f"  {shape}: lower={lower*1000:.0f}ms")
            else:
                print(f"  {shape}: FAILED")

        # Calculate ratio
        if len(shapes) == 2 and all(results[s][component]['success'] for s in shapes):
            ratio = results[shapes[1]][component]['lower'] / results[shapes[0]][component]['lower']
            print(f"  Ratio (256,2048)/(16,2048): {ratio:.2f}x")


if __name__ == "__main__":
    main()
