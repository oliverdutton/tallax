"""Separate tracing time from C++ lowering time with detailed instrumentation."""

import time
import jax
import jax.numpy as jnp
from functools import partial
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import sys

from tallax.tax.divide_and_filter_topk.topk import binned_topk, bitonic_topk_arrays
from tallax.tax.utils import ceil_multiple, log2, to_32bit_dtype, get_dtype_info


# Global timing state
timing_events = []


def log_event(event_name):
    """Log a timing event."""
    timing_events.append((event_name, time.perf_counter()))
    print(f"[{time.perf_counter():.3f}] {event_name}", flush=True)


def kernel_simplified(
    logits_ref, k_vmem_ref, topk_vals_ref, topk_idxs_ref, valid_ref,
    max_depth_ref, cutoff_vals_ref, bins_topm_vals_ref, bins_topm_idxs_ref,
    termination_flag_ref, *, max_k, num_bins, bins_topm_unroll,
    bins_topm_schedule, replace_val,
):
    """Simplified kernel for timing tests."""
    block_token = logits_ref.shape[0]
    shape = (block_token, bins_topm_vals_ref.shape[1])
    pid = pl.program_id(0)
    token_slice = pl.dslice(pid * block_token, block_token)

    bins_topm_vals_ref[token_slice] = jnp.full(shape, get_dtype_info(logits_ref).min, dtype=bins_topm_vals_ref.dtype)
    for i in range(block_token):
        max_depth_ref[pid * block_token + i] = max_k
    termination_flag_ref[0] = 0

    for completed_m, m in zip(bins_topm_schedule, bins_topm_schedule[1:]):
        @pl.when(termination_flag_ref[0] == 0)
        def _():
            bins_topm_vals, bins_topm_idxs = binned_topk(
                logits_ref, k=m,
                bins_topk_vals=[bins_topm_vals_ref[token_slice, pl.dslice(i * num_bins, num_bins)].astype(to_32bit_dtype(logits_ref.dtype)) for i in range(m)],
                bins_topk_idxs=[bins_topm_idxs_ref[token_slice, pl.dslice(i * num_bins, num_bins)] for i in range(m)],
                num_bins=num_bins, completed_k=completed_m, unroll=bins_topm_unroll,
            )

            for i in range(completed_m, m):
                bins_topm_vals_ref[token_slice, pl.dslice(i * num_bins, num_bins)] = bins_topm_vals[i].astype(bins_topm_vals_ref.dtype)
                bins_topm_idxs_ref[token_slice, pl.dslice(i * num_bins, num_bins)] = bins_topm_idxs[i].astype(bins_topm_idxs_ref.dtype)
            if m >= max_k or m == 1:
                return

            pivot = bins_topm_vals[m - 1].max(-1, keepdims=True)
            num_larger = sum((v >= pivot) for v in bins_topm_vals[: m - 1]).astype(to_32bit_dtype(logits_ref.dtype)).sum(-1)

            termination_flag_ref[0] = 0
            for i in range(block_token):
                token_idx = pid * block_token + i
                contains_topk = num_larger[i] >= k_vmem_ref[token_idx]
                termination_flag_ref[0] += contains_topk
                current_max = max_depth_ref[token_idx]
                max_depth_ref[token_idx] = jnp.where(contains_topk & (current_max == max_k), m - 1, current_max)
                cutoff_vals_ref[token_idx] = pivot.squeeze(1)[i]

            @pl.when(termination_flag_ref[0] != block_token)
            def _():
                termination_flag_ref[0] = 0

    @pl.when(pl.program_id(0) == (pl.num_programs(0) - 1))
    def _():
        depth_upper = bins_topm_schedule[-1]
        vals_input = bins_topm_vals_ref[:, : depth_upper * num_bins]
        idxs_input = bins_topm_idxs_ref[:, : depth_upper * num_bins]
        vals, idxs = bitonic_topk_arrays([vals_input, idxs_input], num_keys=1, k=max_k)
        topk_vals_ref[...], topk_idxs_ref[...] = vals.astype(topk_vals_ref.dtype), idxs
        valid_ref[0] = 1


def create_pallas_fn(shape=(16, 2048)):
    """Create pallas function."""
    num_tokens, vocab_size = shape
    num_tokens_padded = ceil_multiple(num_tokens, 8)
    max_k = 128
    num_bins = 256
    bins_topm_schedule = (0, 5, 9)
    max_m = bins_topm_schedule[-1]
    buffer_size = max(max_m, 2 ** log2(max_m - 1)) * num_bins

    key = jax.random.PRNGKey(42)
    logits = jax.random.normal(key, shape).astype(jnp.bfloat16)
    k = jnp.full((num_tokens,), 64, dtype=jnp.int32)

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

    def pallas_fn():
        return pl.pallas_call(
            partial(kernel_simplified, max_k=max_k, num_bins=num_bins, bins_topm_unroll=64,
                    bins_topm_schedule=bins_topm_schedule, replace_val=-1e12),
            in_specs=(pl.BlockSpec((8, vocab_size), lambda i: (i, 0)), pl.BlockSpec(memory_space=pltpu.VMEM)),
            out_shape=output_shapes, scratch_shapes=tuple(scratch_shapes),
            grid=(pl.cdiv(num_tokens, 8),), out_specs=output_specs,
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=int(0.9 * 2**27)),
            interpret=False, backend='mosaic_tpu',
        )(logits, k)

    return pallas_fn


def instrument_jax_tracing():
    """Patch JAX to log when tracing happens."""
    import jax._src.core as core

    # Store original functions
    original_trace_to_jaxpr = None

    try:
        # Try to find the trace_to_jaxpr function
        from jax._src import pallas as pallas_src
        if hasattr(pallas_src, 'pallas_call'):
            from jax._src.pallas import pallas_call as pallas_call_mod
            if hasattr(pallas_call_mod, '_trace_kernel_to_jaxpr'):
                original_trace_to_jaxpr = pallas_call_mod._trace_kernel_to_jaxpr

                def traced_trace_to_jaxpr(*args, **kwargs):
                    log_event("  → TRACING START (Python -> Jaxpr)")
                    result = original_trace_to_jaxpr(*args, **kwargs)
                    log_event("  → TRACING END (Jaxpr created)")
                    return result

                pallas_call_mod._trace_kernel_to_jaxpr = traced_trace_to_jaxpr
                return True
    except Exception as e:
        print(f"Could not instrument tracing: {e}")

    return False


def test_timing_breakdown(shape=(16, 2048)):
    """Test with detailed timing breakdown."""
    global timing_events
    timing_events = []

    print("="*70)
    print(f"Tracing vs Lowering Time Breakdown - {shape}")
    print("="*70)
    print()

    # Try to instrument JAX
    instrumented = instrument_jax_tracing()
    if instrumented:
        print("✓ JAX tracing instrumented - will see TRACING START/END\n")
    else:
        print("✗ Could not instrument JAX - will only see overall timing\n")

    log_event("START: Creating pallas function")
    pallas_fn = create_pallas_fn(shape)

    log_event("START: jax.jit()")
    jitted = jax.jit(pallas_fn)
    log_event("END: jax.jit() (instant)")

    print()
    log_event("START: jitted.lower() - FULL LOWERING")
    print("  (This includes tracing + C++ compilation)")
    print()

    t_lower_start = time.perf_counter()
    lowered = jitted.lower()
    t_lower_end = time.perf_counter()

    print()
    log_event("END: jitted.lower() COMPLETE")

    log_event("START: HLO extraction")
    t_hlo_start = time.perf_counter()
    hlo = lowered.as_text()
    t_hlo_end = time.perf_counter()
    log_event("END: HLO extraction")

    # Analyze timing events
    print("\n" + "="*70)
    print("TIMING ANALYSIS")
    print("="*70)

    # Find tracing events if instrumented
    tracing_start = None
    tracing_end = None
    for event, t in timing_events:
        if "TRACING START" in event:
            tracing_start = t
        elif "TRACING END" in event:
            tracing_end = t

    lower_total = t_lower_end - t_lower_start
    hlo_time = t_hlo_end - t_hlo_start

    print(f"\nTotal .lower() time:        {lower_total:7.2f}s")

    if tracing_start and tracing_end:
        tracing_time = tracing_end - tracing_start
        cpp_lowering = lower_total - tracing_time

        print(f"\nBreakdown:")
        print(f"  Python tracing:           {tracing_time:7.2f}s ({tracing_time/lower_total*100:5.1f}%)")
        print(f"  C++ lowering:             {cpp_lowering:7.2f}s ({cpp_lowering/lower_total*100:5.1f}%)")
    else:
        print("\n(Could not separate tracing from C++ lowering)")
        print("Entire .lower() call includes both tracing + C++ compilation")

    print(f"\nHLO extraction:             {hlo_time:7.2f}s")
    print(f"HLO size:                   {len(hlo):,} chars")

    print("\n" + "="*70)
    print("EVENT LOG")
    print("="*70)

    if timing_events:
        t_start = timing_events[0][1]
        prev_t = t_start

        for event, t in timing_events:
            elapsed_total = t - t_start
            elapsed_since_prev = t - prev_t
            print(f"[+{elapsed_total:7.2f}s] (+{elapsed_since_prev:6.3f}s) {event}")
            prev_t = t

    return {
        'total_lower': lower_total,
        'tracing': tracing_end - tracing_start if (tracing_start and tracing_end) else None,
        'hlo_time': hlo_time
    }


def main():
    print("Testing tracing vs lowering time separation\n")

    # Test on (256, 2048) for full analysis
    result = test_timing_breakdown(shape=(256, 2048))

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if result['tracing'] is not None:
        print(f"\nPython tracing:    {result['tracing']:.2f}s")
        print(f"C++ lowering:      {result['total_lower'] - result['tracing']:.2f}s")
        print(f"Total:             {result['total_lower']:.2f}s")

        tracing_pct = result['tracing'] / result['total_lower'] * 100
        cpp_pct = 100 - tracing_pct

        print(f"\nTracing is {tracing_pct:.1f}% of total lowering time")
        print(f"C++ compilation is {cpp_pct:.1f}% of total lowering time")
    else:
        print(f"\nTotal .lower():    {result['total_lower']:.2f}s")
        print("(Tracing + C++ compilation combined)")


if __name__ == "__main__":
    main()
