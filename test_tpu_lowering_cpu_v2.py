"""Test TPU lowering on CPU using backend parameter."""

import time
import jax
import jax.numpy as jnp
import numpy as np
from tallax.vllm import topk_topp_and_sample
from tallax.vllm.tpu_inference_sampling_as_standalone_file import TPUSupportedSamplingMetadata
from tallax.tax import bitonic_top_k
from tallax.tax.divide_and_filter_topk.topk import _top_bounded_k


def time_lowering_with_backend(fn_body, *args, name=None, backend='tpu', **kwargs):
    """Time lowering with explicit backend specification.

    Args:
        fn_body: The function to lower (not jitted yet)
        *args: Arguments to pass to the function
        name: Optional name for display
        backend: Backend to target ('tpu', 'cpu', 'gpu')
        **kwargs: Keyword arguments to pass to the function
    """
    func_name = name or getattr(fn_body, '__name__', 'unnamed_function')

    print(f"\n{'='*60}")
    print(f"Lowering: {func_name}")
    print(f"Backend: {backend}")
    print(f"{'='*60}")

    # Time lowering
    t0 = time.perf_counter()
    try:
        # Wrap in jit with explicit backend
        jitted = jax.jit(fn_body, backend=backend)
        lowered = jitted.lower(*args, **kwargs)

        t1 = time.perf_counter()
        lower_time = t1 - t0
        print(f"  Lowering time: {lower_time*1000:.2f} ms")
        print(f"  ✓ SUCCESS")
        success = True

        # Try to get HLO info
        try:
            hlo_text = lowered.as_text()
            print(f"  HLO size: {len(hlo_text):,} characters")
            # Count number of instructions
            num_instructions = hlo_text.count('\n')
            print(f"  HLO lines: {num_instructions:,}")
        except:
            print(f"  HLO: (unavailable)")

    except Exception as e:
        t1 = time.perf_counter()
        lower_time = t1 - t0
        print(f"  Lowering time: {lower_time*1000:.2f} ms")
        print(f"  ✗ FAILED")
        error_msg = str(e)
        if len(error_msg) > 300:
            error_msg = error_msg[:300] + "..."
        print(f"  Error: {type(e).__name__}")
        print(f"  {error_msg}")
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

    # Test top_bounded_k (the main component)
    print("\n--- Test: _top_bounded_k (divide-and-filter) ---")
    timings_bounded = time_lowering_with_backend(
        _top_bounded_k,
        logits,
        tpu_sampling_metadata.top_k,
        max_k=128,
        block_token=8,
        num_bins=256,
        bins_topm_schedule=(0, 5, 9),
        bins_topm_unroll=64,
        guarantee_convergence=True,
        replace_val=-1e12,
        interpret=False,
        name="_top_bounded_k",
        backend='tpu'
    )

    return {
        'bounded': timings_bounded,
    }


def main():
    print("=" * 70)
    print("TPU Lowering on CPU - Using backend='tpu'")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Default backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")

    # Test both shapes
    shapes = [(16, 2048), (256, 2048)]
    results = {}

    for shape in shapes:
        results[shape] = test_shape_lowering(shape)

    # Summary comparison
    print("\n" + "=" * 70)
    print("LOWERING TIME COMPARISON")
    print("=" * 70)

    print(f"\n_TOP_BOUNDED_K:")
    times = []
    for shape in shapes:
        if results[shape]['bounded']['success']:
            lower_ms = results[shape]['bounded']['lower'] * 1000
            print(f"  {shape}: {lower_ms:.0f} ms")
            times.append(results[shape]['bounded']['lower'])
        else:
            print(f"  {shape}: FAILED")

    if len(times) == 2:
        ratio = times[1] / times[0]
        print(f"\n  → Ratio (256,2048)/(16,2048): {ratio:.2f}x")

    print("\n" + "=" * 70)
    print("EXPECTED: Lowering time should scale with buffer size")
    print("If (256,2048) is ~13-16x slower, buffer size is the bottleneck")
    print("=" * 70)


if __name__ == "__main__":
    main()
