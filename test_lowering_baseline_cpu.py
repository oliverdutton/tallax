"""Test TPU lowering time on CPU with the modified JAX branch.

This tests ONLY lowering time, not compilation.
Uses backend='mosaic_tpu' in pallas_call, which should now work on CPU.
"""

import time
import jax
import jax.numpy as jnp
from functools import partial
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax.vllm.tpu_inference_sampling_as_standalone_file import TPUSupportedSamplingMetadata
from tallax.tax.divide_and_filter_topk.topk import _top_bounded_k
from tallax.tax.utils import ceil_multiple, log2, to_32bit_dtype


def time_lowering_only(fn_callable, name="function"):
    """Time only the lowering stage for a pallas_call function."""

    print(f"\n{'='*60}")
    print(f"Lowering: {name}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    try:
        # Just jit it normally - the pallas_call has backend='mosaic_tpu'
        jitted = jax.jit(fn_callable)

        # Lower it - this should now work on CPU!
        lowered = jitted.lower()

        t1 = time.perf_counter()
        lower_time = t1 - t0

        print(f"  ✓ Lowering successful: {lower_time*1000:.2f} ms")

        # Try to get HLO size
        try:
            hlo_text = lowered.as_text()
            print(f"  HLO size: {len(hlo_text):,} characters")
            num_lines = hlo_text.count('\n')
            print(f"  HLO lines: {num_lines:,}")
        except:
            print(f"  HLO: (unavailable)")

        return {'time': lower_time, 'success': True}

    except Exception as e:
        t1 = time.perf_counter()
        lower_time = t1 - t0

        print(f"  ✗ Lowering failed: {lower_time*1000:.2f} ms")
        print(f"  Error: {type(e).__name__}")
        error_msg = str(e)
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "..."
        print(f"  {error_msg}")

        return {'time': lower_time, 'success': False}


def create_pallas_fn(logits, k, max_k=128, block_token=8, num_bins=256,
                     bins_topm_schedule=(0, 5, 9), bins_topm_unroll=64):
    """Create a pallas_call function with backend='mosaic_tpu'."""

    num_tokens, vocab_size = logits.shape
    num_tokens_padded = ceil_multiple(num_tokens, block_token)
    max_m = bins_topm_schedule[-1]
    buffer_size = max(max_m, 2 ** log2(max_m - 1)) * num_bins

    output_shapes = (
        jax.ShapeDtypeStruct((num_tokens, max_k), logits.dtype),
        jax.ShapeDtypeStruct((num_tokens, max_k), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
        jax.ShapeDtypeStruct((num_tokens_padded,), jnp.int32),
        jax.ShapeDtypeStruct((num_tokens_padded,), to_32bit_dtype(logits.dtype)),
    )

    output_specs = (
        pl.BlockSpec((num_tokens_padded, max_k), lambda i: (0, 0)),
        pl.BlockSpec((num_tokens_padded, max_k), lambda i: (0, 0)),
        pl.BlockSpec(memory_space=pltpu.SMEM),
        pl.BlockSpec(memory_space=pltpu.SMEM),
        pl.BlockSpec(memory_space=pltpu.SMEM),
    )

    scratch_shapes = [
        pltpu.VMEM((num_tokens_padded, buffer_size), to_32bit_dtype(logits.dtype)),
        pltpu.VMEM((num_tokens_padded, buffer_size), jnp.int32),
        pltpu.SMEM((1,), jnp.int32),
    ]

    from tallax.tax.divide_and_filter_topk.topk import dynamic_topk_refs

    # Create pallas_call with backend='mosaic_tpu'
    def pallas_fn():
        return pl.pallas_call(
            partial(
                dynamic_topk_refs,
                max_k=max_k,
                num_bins=num_bins,
                bins_topm_unroll=bins_topm_unroll,
                bins_topm_schedule=bins_topm_schedule,
                guarantee_convergence=True,
                replace_val=-1e12,
            ),
            in_specs=(
                pl.BlockSpec((block_token, vocab_size), lambda i: (i, 0)),
                pl.BlockSpec(memory_space=pltpu.SMEM),
                pl.BlockSpec(memory_space=pltpu.SMEM),
            ),
            out_shape=output_shapes,
            scratch_shapes=tuple(scratch_shapes),
            grid=(pl.cdiv(num_tokens, block_token),),
            out_specs=output_specs,
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=int(0.9 * 2**27)),
            interpret=False,
            backend='mosaic_tpu',  # KEY: This should now work on CPU
        )(logits, k, k)

    return pallas_fn


def test_baseline_lowering(shape, dtype=jnp.bfloat16, seed=42):
    """Test baseline lowering time without optimization barriers."""

    num_tokens, vocab_size = shape

    print(f"\n{'#'*70}")
    print(f"Testing BASELINE: shape={shape}")
    print(f"{'#'*70}")

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

    # Create the pallas function
    pallas_fn = create_pallas_fn(logits, k, max_k=128, bins_topm_unroll=64)

    # Time lowering
    result = time_lowering_only(pallas_fn, name=f"top_bounded_k_{shape}")

    return result


def main():
    print("="*70)
    print("TPU Lowering Time Test on CPU")
    print("="*70)
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Using backend='mosaic_tpu' in pallas_call")
    print()

    # Test both shapes
    shapes = [(16, 2048), (256, 2048)]
    results = {}

    for shape in shapes:
        # Clear caches between tests
        jax.clear_caches()
        results[shape] = test_baseline_lowering(shape)

    # Summary
    print("\n" + "="*70)
    print("LOWERING TIME COMPARISON")
    print("="*70)

    for shape in shapes:
        if results[shape]['success']:
            print(f"{shape}: {results[shape]['time']*1000:.0f} ms")
        else:
            print(f"{shape}: FAILED")

    if all(results[s]['success'] for s in shapes):
        ratio = results[shapes[1]]['time'] / results[shapes[0]]['time']
        print(f"\nRatio (256,2048)/(16,2048): {ratio:.2f}x")
        print("\nExpected from your data:")
        print("  (16, 2048): ~6.6s lowering")
        print("  (256, 2048): ~87.2s lowering")
        print("  Expected ratio: ~13.2x")

        if ratio < 10:
            print("\n✓ Lowering scales better than expected!")
        elif ratio < 15:
            print("\n→ Lowering scales as expected (~13x)")
        else:
            print("\n✗ Lowering scales worse than expected")


if __name__ == "__main__":
    main()
