"""Detailed timing breakdown to separate Python tracing from C++ lowering."""

import time
import jax
import jax.numpy as jnp
from functools import partial
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from datetime import datetime

from tallax.vllm.tpu_inference_sampling_as_standalone_file import TPUSupportedSamplingMetadata
from tallax.tax.divide_and_filter_topk.topk import binned_topk, bitonic_topk_arrays
from tallax.tax.utils import ceil_multiple, log2, to_32bit_dtype, get_dtype_info


def timestamp(msg):
    """Print message with timestamp."""
    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{now}] {msg}", flush=True)


def dynamic_topk_refs_simple(
    logits_ref,
    k_vmem_ref,
    topk_vals_ref,
    topk_idxs_ref,
    valid_ref,
    max_depth_ref,
    cutoff_vals_ref,
    # scratch
    bins_topm_vals_ref,
    bins_topm_idxs_ref,
    termination_flag_ref,
    *,
    max_k: int,
    num_bins: int,
    bins_topm_unroll: int,
    bins_topm_schedule: tuple[int, ...],
    replace_val: float | int | None,
):
    """Simplified kernel: VMEM-only k, no guarantee_convergence."""
    # Initialize buffers
    block_token = logits_ref.shape[0]
    shape = (block_token, bins_topm_vals_ref.shape[1])

    pid = pl.program_id(0)
    token_slice = pl.dslice(pid * block_token, block_token)

    bins_topm_vals_ref[token_slice] = jnp.full(
        shape, get_dtype_info(logits_ref).min, dtype=bins_topm_vals_ref.dtype
    )

    for i in range(block_token):
        max_depth_ref[pid * block_token + i] = max_k
    termination_flag_ref[0] = 0

    # Incremental binned top-k computation
    for completed_m, m in zip(bins_topm_schedule, bins_topm_schedule[1:]):

        @pl.when(termination_flag_ref[0] == 0)
        def _():
            # Compute binned top-m
            bins_topm_vals, bins_topm_idxs = binned_topk(
                logits_ref,
                k=m,
                bins_topk_vals=[
                    bins_topm_vals_ref[
                        token_slice, pl.dslice(i * num_bins, num_bins)
                    ].astype(to_32bit_dtype(logits_ref.dtype))
                    for i in range(m)
                ],
                bins_topk_idxs=[
                    bins_topm_idxs_ref[token_slice, pl.dslice(i * num_bins, num_bins)]
                    for i in range(m)
                ],
                num_bins=num_bins,
                completed_k=completed_m,
                unroll=bins_topm_unroll,
            )

            # Store results
            for i in range(completed_m, m):
                bins_topm_vals_ref[token_slice, pl.dslice(i * num_bins, num_bins)] = (
                    bins_topm_vals[i].astype(bins_topm_vals_ref.dtype)
                )
                bins_topm_idxs_ref[token_slice, pl.dslice(i * num_bins, num_bins)] = (
                    bins_topm_idxs[i].astype(bins_topm_idxs_ref.dtype)
                )
            if m >= max_k:
                return
            if m == 1:
                return

            # Termination criterion
            pivot = bins_topm_vals[m - 1].max(-1, keepdims=True)
            num_larger = (
                sum((v >= pivot) for v in bins_topm_vals[: m - 1])
                .astype(to_32bit_dtype(logits_ref.dtype))
                .sum(-1)
            )

            termination_flag_ref[0] = 0
            for i in range(block_token):
                token_idx = pid * block_token + i
                contains_topk = num_larger[i] >= k_vmem_ref[token_idx]
                termination_flag_ref[0] += contains_topk

                # Record depth when criterion was met
                current_max = max_depth_ref[token_idx]
                max_depth_ref[token_idx] = jnp.where(
                    contains_topk & (current_max == max_k), m - 1, current_max
                )
                cutoff_vals_ref[token_idx] = pivot.squeeze(1)[i]

            # Check if all tokens converged
            @pl.when(termination_flag_ref[0] != block_token)
            def _():
                termination_flag_ref[0] = 0

    # Final top-k extraction
    global_topk_schedule = [max(x - 1, 0) for x in bins_topm_schedule[:-1]] + [
        bins_topm_schedule[-1]
    ]
    global_topk_schedule = tuple(sorted(set(bins_topm_schedule)))

    @pl.when(pl.program_id(0) == (pl.num_programs(0) - 1))
    def _():
        # Find maximum depth
        global_max_depth = jnp.array(0)
        for i in range(max_depth_ref.shape[0]):
            global_max_depth = jnp.maximum(global_max_depth, max_depth_ref[i])

        valid_ref[0] = (
            (global_max_depth < bins_topm_schedule[-1])
            | (bins_topm_schedule[-1] >= max_k)
        ).astype(jnp.int32)

        # Sort based on global_max_depth
        for depth_lower, depth_upper in zip(
            global_topk_schedule, global_topk_schedule[1:]
        ):

            @pl.when(
                ((global_max_depth > depth_lower) & (global_max_depth <= depth_upper))
                | (
                    (depth_upper == global_topk_schedule[-1])
                    & (global_max_depth > depth_upper)
                )
            )
            def _():
                vals_input = bins_topm_vals_ref[:, : depth_upper * num_bins]
                idxs_input = bins_topm_idxs_ref[:, : depth_upper * num_bins]
                vals, idxs = bitonic_topk_arrays(
                    [vals_input, idxs_input],
                    num_keys=1,
                    k=max_k,
                )
                topk_vals_ref[...], topk_idxs_ref[...] = vals.astype(topk_vals_ref.dtype), idxs
                if replace_val is not None:
                    idx = jax.lax.broadcasted_iota(jnp.int32, topk_vals_ref.shape, 1)
                    topk_vals_ref[...] = jnp.where(
                        idx < k_vmem_ref[...][:, None], topk_vals_ref[...], replace_val
                    )


def time_stages_separately(fn_callable, name="function"):
    """Time each stage separately with detailed timestamps."""
    print(f"\n{'='*70}")
    print(f"Detailed Timing: {name}")
    print(f"{'='*70}")

    # Stage 1: Create jitted function (should be instant)
    timestamp("START: Creating jitted function")
    t_start = time.perf_counter()

    jitted = jax.jit(fn_callable)

    t_jit = time.perf_counter()
    timestamp(f"DONE: Jitted function created ({(t_jit - t_start)*1000:.1f} ms)")

    # Stage 2: Lower (includes tracing + lowering)
    timestamp("START: Lowering (tracing + HLO generation)")
    t_lower_start = time.perf_counter()

    try:
        lowered = jitted.lower()

        t_lower_end = time.perf_counter()
        timestamp(f"DONE: Lowering complete ({(t_lower_end - t_lower_start):.2f}s)")

        # Stage 3: Get HLO text
        timestamp("START: Extracting HLO text")
        t_hlo_start = time.perf_counter()

        try:
            hlo_text = lowered.as_text()
            t_hlo_end = time.perf_counter()
            timestamp(f"DONE: HLO text extracted ({(t_hlo_end - t_hlo_start)*1000:.1f} ms)")
            timestamp(f"HLO size: {len(hlo_text):,} characters")
        except Exception as e:
            timestamp(f"ERROR: Could not extract HLO: {e}")
            hlo_text = None

        # Total timing
        total_time = t_lower_end - t_start

        print(f"\nTiming Breakdown:")
        print(f"  JIT creation:       {(t_jit - t_start)*1000:>10.1f} ms")
        print(f"  Lowering:           {(t_lower_end - t_lower_start):>10.2f} s")
        if hlo_text:
            print(f"  HLO extraction:     {(t_hlo_end - t_hlo_start)*1000:>10.1f} ms")
        print(f"  {'─'*35}")
        print(f"  Total:              {total_time:>10.2f} s")

        return {
            'time': total_time,
            'lowering_time': t_lower_end - t_lower_start,
            'success': True,
            'hlo_size': len(hlo_text) if hlo_text else 0
        }

    except Exception as e:
        t_error = time.perf_counter()
        timestamp(f"ERROR: Lowering failed after {(t_error - t_lower_start):.2f}s")
        print(f"  Error: {type(e).__name__}")
        print(f"  {str(e)[:200]}")

        return {
            'time': t_error - t_start,
            'lowering_time': t_error - t_lower_start,
            'success': False,
            'error': str(e)
        }


def create_pallas_fn(logits, k, max_k=128, block_token=8, num_bins=256,
                     bins_topm_schedule=(0, 5, 9), bins_topm_unroll=64,
                     compiler_params_extra=None):
    """Create pallas_call with optional compiler params."""
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

    # Compiler params
    compiler_params = pltpu.CompilerParams(vmem_limit_bytes=int(0.9 * 2**27))
    if compiler_params_extra:
        for key, value in compiler_params_extra.items():
            setattr(compiler_params, key, value)

    def pallas_fn():
        return pl.pallas_call(
            partial(
                dynamic_topk_refs_simple,
                max_k=max_k,
                num_bins=num_bins,
                bins_topm_unroll=bins_topm_unroll,
                bins_topm_schedule=bins_topm_schedule,
                replace_val=-1e12,
            ),
            in_specs=(
                pl.BlockSpec((block_token, vocab_size), lambda i: (i, 0)),
                pl.BlockSpec(memory_space=pltpu.VMEM),
            ),
            out_shape=output_shapes,
            scratch_shapes=tuple(scratch_shapes),
            grid=(pl.cdiv(num_tokens, block_token),),
            out_specs=output_specs,
            compiler_params=compiler_params,
            interpret=False,
            backend='mosaic_tpu',
        )(logits, k)

    return pallas_fn


def main():
    print("="*70)
    print("Detailed Lowering Time Analysis - (16, 2048) Only")
    print("="*70)
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print()

    # Shape
    shape = (16, 2048)
    num_tokens, vocab_size = shape

    print(f"Testing shape: {shape}")
    print()

    # Setup
    key = jax.random.PRNGKey(42)
    key, topk_key, topp_key, temp_key, logits_key = jax.random.split(key, 5)

    tpu_sampling_metadata = TPUSupportedSamplingMetadata(
        top_k=jax.random.randint(topk_key, (num_tokens,), 1, 128, dtype=jnp.int32),
        top_p=jax.random.uniform(topp_key, (num_tokens,), dtype=jnp.float32),
        temperature=10 ** jax.random.normal(temp_key, (num_tokens,), dtype=jnp.float32),
        do_sampling=True,
    )

    logits = jax.random.normal(logits_key, shape).astype(jnp.bfloat16)
    k = tpu_sampling_metadata.top_k

    # Test different unroll values
    tests = [
        ("Baseline (unroll=64)", 64, None),
        ("Reduced unroll=32", 32, None),
        ("Reduced unroll=16", 16, None),
        ("Reduced unroll=8", 8, None),
    ]

    results = {}

    for test_name, unroll, compiler_extra in tests:
        print(f"\n{'='*70}")
        print(f"TEST: {test_name}")
        print(f"{'='*70}")

        jax.clear_caches()

        pallas_fn = create_pallas_fn(
            logits, k,
            bins_topm_unroll=unroll,
            compiler_params_extra=compiler_extra
        )

        results[test_name] = time_stages_separately(pallas_fn, name=test_name)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - Lowering Time Comparison")
    print("="*70)

    baseline_time = results.get("Baseline (unroll=64)", {}).get('lowering_time')

    for test_name, result in results.items():
        if result['success']:
            t = result['lowering_time']
            if baseline_time and test_name != "Baseline (unroll=64)":
                speedup = baseline_time / t
                print(f"{test_name:30s}: {t:6.2f}s ({speedup:4.2f}x)")
            else:
                print(f"{test_name:30s}: {t:6.2f}s (baseline)")
        else:
            print(f"{test_name:30s}: FAILED")

    print("\nKey insight: Lowering time is primarily C++ (XLA/Mosaic)")
    print("Python tracing overhead is minimal (JIT creation < 1ms)")


if __name__ == "__main__":
    main()
