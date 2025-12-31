"""Debug script to time JAX compilation stages for topk/topp/sample operations."""

import time
import jax
import jax.numpy as jnp
import numpy as np
from tallax.vllm import topk_topp_and_sample, top_p_and_sample
from tallax.vllm.tpu_inference_sampling_as_standalone_file import TPUSupportedSamplingMetadata
from tallax.tax import bitonic_top_k
from tallax.tax.divide_and_filter_topk.topk import top_bounded_k


def time_jax_compilation_stages(fn, *args, name=None, **kwargs):
    """Time JAX compilation stages (jaxpr, lowering, compilation)."""
    func_name = name or getattr(fn, '__name__', 'unnamed_function')

    print(f"\n{'='*60}")
    print(f"Timing compilation stages for: {func_name}")
    print(f"{'='*60}")

    # Stage 1: Create jaxpr (trace through the function)
    try:
        t0 = time.perf_counter()
        if hasattr(fn, '_fun'):
            inner_func = fn._fun
        else:
            inner_func = fn
        jaxpr = jax.make_jaxpr(inner_func)(*args, **kwargs)
        t1 = time.perf_counter()
        jaxpr_time = t1 - t0
        print(f"  jaxpr creation: {jaxpr_time*1000:.2f} ms")
    except Exception as e:
        jaxpr_time = -1
        print(f'  jaxpr creation failed: {e}')

    # Stage 2: Lower to StableHLO
    @jax.jit
    def jitted_func():
        return fn(*args, **kwargs)

    t0 = time.perf_counter()
    lowered = jitted_func.lower()
    t1 = time.perf_counter()
    lower_time = t1 - t0
    print(f"  lowering:       {lower_time*1000:.2f} ms")

    # Stage 3: Compile
    t0 = time.perf_counter()
    compiled = lowered.compile()
    t1 = time.perf_counter()
    compile_time = t1 - t0
    print(f"  compilation:    {compile_time*1000:.2f} ms")

    total_time = lower_time + compile_time
    print(f"  {'─'*40}")
    print(f"  TOTAL:          {total_time*1000:.2f} ms")
    print(f"{'='*60}\n")

    # Execute the compiled function to verify it works
    result = compiled()

    timings = {
        'jaxpr': jaxpr_time,
        'lower': lower_time,
        'compile': compile_time,
        'total': total_time,
    }

    return result, timings


def test_shape_compilation(shape, dtype=jnp.bfloat16, seed=42):
    """Test compilation timings for a specific shape."""
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

    # Test individual components
    print("\n--- Component 1: top_bounded_k (divide-and-filter topk) ---")
    _, timings_bounded = time_jax_compilation_stages(
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

    # Get the topk results to use in next step
    topk_logits, topk_idxs = top_bounded_k(
        logits,
        k=tpu_sampling_metadata.top_k,
        max_k=128,
        num_bins=256,
        bins_topm_schedule=(5, 9),
        guarantee_convergence=True,
        replace_val=-1e12,
    )

    print("\n--- Component 2: top_p_and_sample ---")
    _, timings_sample = time_jax_compilation_stages(
        top_p_and_sample,
        topk_logits,
        topk_idxs,
        sample_key,
        top_p=tpu_sampling_metadata.top_p,
        temperature=tpu_sampling_metadata.temperature,
        vocab_size=vocab_size,
        replace_val=-1e12,
        name="top_p_and_sample"
    )

    print("\n--- Full Pipeline: topk_topp_and_sample ---")
    _, timings_full = time_jax_compilation_stages(
        topk_topp_and_sample,
        sample_key,
        logits,
        tpu_sampling_metadata,
        name="topk_topp_and_sample"
    )

    print("\n--- For reference: bitonic_top_k ---")
    _, timings_bitonic = time_jax_compilation_stages(
        bitonic_top_k,
        logits,
        k=128,
        name="bitonic_top_k"
    )

    return {
        'bounded': timings_bounded,
        'sample': timings_sample,
        'full': timings_full,
        'bitonic': timings_bitonic,
    }


def main():
    print("=" * 70)
    print("JAX Compilation Timing Debug")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")

    # Test both shapes
    shapes = [(16, 2048), (256, 2048)]
    results = {}

    for shape in shapes:
        results[shape] = test_shape_compilation(shape)

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    for component in ['bounded', 'sample', 'full', 'bitonic']:
        print(f"\n{component.upper()}:")
        for shape in shapes:
            total = results[shape][component]['total']
            lower = results[shape][component]['lower']
            compile = results[shape][component]['compile']
            print(f"  {shape}: total={total*1000:.0f}ms (lower={lower*1000:.0f}ms, compile={compile*1000:.0f}ms)")

        # Calculate ratio
        if len(shapes) == 2:
            ratio = results[shapes[1]][component]['total'] / results[shapes[0]][component]['total']
            print(f"  Ratio (256,2048)/(16,2048): {ratio:.2f}x")


if __name__ == "__main__":
    main()
