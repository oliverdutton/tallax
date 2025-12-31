"""Detailed timestamps to verify when lowering actually happens."""

import time
import jax
import jax.numpy as jnp
from functools import partial
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import sys

from tallax.vllm.tpu_inference_sampling_as_standalone_file import TPUSupportedSamplingMetadata
from tallax.tax.divide_and_filter_topk.topk import binned_topk, bitonic_topk_arrays
from tallax.tax.utils import ceil_multiple, log2, to_32bit_dtype, get_dtype_info


def ts():
    """Return current timestamp in seconds since start."""
    return time.perf_counter()


def print_ts(msg, t_start):
    """Print message with elapsed time."""
    elapsed = ts() - t_start
    print(f"[{elapsed:7.3f}s] {msg}", flush=True)
    sys.stdout.flush()


def dynamic_topk_refs_instrumented(
    logits_ref,
    k_vmem_ref,
    topk_vals_ref,
    topk_idxs_ref,
    valid_ref,
    max_depth_ref,
    cutoff_vals_ref,
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
    """Kernel with timestamps at each major section."""
    block_token = logits_ref.shape[0]
    shape = (block_token, bins_topm_vals_ref.shape[1])
    pid = pl.program_id(0)
    token_slice = pl.dslice(pid * block_token, block_token)

    # Section 1: Init
    bins_topm_vals_ref[token_slice] = jnp.full(
        shape, get_dtype_info(logits_ref).min, dtype=bins_topm_vals_ref.dtype
    )
    for i in range(block_token):
        max_depth_ref[pid * block_token + i] = max_k
    termination_flag_ref[0] = 0

    # Section 2: Incremental binned topk
    for completed_m, m in zip(bins_topm_schedule, bins_topm_schedule[1:]):
        @pl.when(termination_flag_ref[0] == 0)
        def _():
            bins_topm_vals, bins_topm_idxs = binned_topk(
                logits_ref,
                k=m,
                bins_topk_vals=[
                    bins_topm_vals_ref[token_slice, pl.dslice(i * num_bins, num_bins)].astype(to_32bit_dtype(logits_ref.dtype))
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
                current_max = max_depth_ref[token_idx]
                max_depth_ref[token_idx] = jnp.where(
                    contains_topk & (current_max == max_k), m - 1, current_max
                )
                cutoff_vals_ref[token_idx] = pivot.squeeze(1)[i]

            @pl.when(termination_flag_ref[0] != block_token)
            def _():
                termination_flag_ref[0] = 0

    # Section 3: Final extraction
    global_topk_schedule = [max(x - 1, 0) for x in bins_topm_schedule[:-1]] + [bins_topm_schedule[-1]]
    global_topk_schedule = tuple(sorted(set(bins_topm_schedule)))

    @pl.when(pl.program_id(0) == (pl.num_programs(0) - 1))
    def _():
        global_max_depth = jnp.array(0)
        for i in range(max_depth_ref.shape[0]):
            global_max_depth = jnp.maximum(global_max_depth, max_depth_ref[i])

        valid_ref[0] = (
            (global_max_depth < bins_topm_schedule[-1])
            | (bins_topm_schedule[-1] >= max_k)
        ).astype(jnp.int32)

        for depth_lower, depth_upper in zip(global_topk_schedule, global_topk_schedule[1:]):
            @pl.when(
                ((global_max_depth > depth_lower) & (global_max_depth <= depth_upper))
                | ((depth_upper == global_topk_schedule[-1]) & (global_max_depth > depth_upper))
            )
            def _():
                vals_input = bins_topm_vals_ref[:, : depth_upper * num_bins]
                idxs_input = bins_topm_idxs_ref[:, : depth_upper * num_bins]
                vals, idxs = bitonic_topk_arrays([vals_input, idxs_input], num_keys=1, k=max_k)
                topk_vals_ref[...], topk_idxs_ref[...] = vals.astype(topk_vals_ref.dtype), idxs
                if replace_val is not None:
                    idx = jax.lax.broadcasted_iota(jnp.int32, topk_vals_ref.shape, 1)
                    topk_vals_ref[...] = jnp.where(
                        idx < k_vmem_ref[...][:, None], topk_vals_ref[...], replace_val
                    )


def create_pallas_fn(logits, k, kernel_fn, max_k=128, block_token=8, num_bins=256,
                     bins_topm_schedule=(0, 5, 9), bins_topm_unroll=64,
                     xla_flags=None):
    """Create pallas_call with optional XLA flags."""
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

    def pallas_fn():
        return pl.pallas_call(
            partial(
                kernel_fn,
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
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=int(0.9 * 2**27)),
            interpret=False,
            backend='mosaic_tpu',
        )(logits, k)

    return pallas_fn


def test_with_timestamps(shape=(16, 2048), xla_flags=None, test_name="baseline"):
    """Test with detailed timestamps at each stage."""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"Shape: {shape}")
    if xla_flags:
        print(f"XLA flags: {xla_flags}")
    print(f"{'='*70}\n")

    t_start = ts()

    # Setup data
    print_ts("START: Setting up test data", t_start)
    num_tokens, vocab_size = shape
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

    print_ts("DONE: Test data ready", t_start)

    # Create pallas function
    print_ts("START: Creating pallas_fn", t_start)
    pallas_fn = create_pallas_fn(logits, k, dynamic_topk_refs_instrumented, xla_flags=xla_flags)
    print_ts("DONE: pallas_fn created", t_start)

    # Create jitted function
    print_ts("START: jax.jit(pallas_fn)", t_start)
    jitted = jax.jit(pallas_fn)
    print_ts("DONE: jax.jit returned", t_start)

    # Lower (this is where the time should be)
    print_ts("START: jitted.lower() - TRACING + LOWERING", t_start)
    lowered = jitted.lower()
    print_ts("DONE: jitted.lower() returned - LOWERING COMPLETE", t_start)

    # Get HLO
    print_ts("START: lowered.as_text()", t_start)
    hlo = lowered.as_text()
    print_ts(f"DONE: HLO extracted ({len(hlo):,} chars)", t_start)

    total = ts() - t_start
    print(f"\nTotal time: {total:.2f}s\n")

    return total


def main():
    print("="*70)
    print("Detailed Timestamp Analysis - Verify Lowering Time")
    print("="*70)
    print(f"JAX version: {jax.__version__}\n")

    # Test 1: Baseline
    jax.clear_caches()
    t1 = test_with_timestamps(shape=(16, 2048), test_name="Baseline")

    # Test 2: Disable XLA optimizations
    print("\n" + "="*70)
    print("Testing with XLA optimizations disabled")
    print("="*70)

    import os

    # Disable many XLA passes
    disabled_flags = {
        'xla_disable_hlo_passes': 'all-reduce-combiner,all-gather-combiner,all-to-all-decomposer,'
                                   'reduce-scatter-combiner,ar-crs-combiner,batch-dot-simplification,'
                                   'algebraic-simplifier,conditional-canonicalizer,tuple-simplifier,'
                                   'while-loop-simplification,gather-simplifier,scatter-simplifier',
    }

    for flag, value in disabled_flags.items():
        os.environ[flag] = value

    jax.clear_caches()
    t2 = test_with_timestamps(shape=(16, 2048), test_name="XLA passes disabled", xla_flags=disabled_flags)

    # Clean up
    for flag in disabled_flags:
        os.environ.pop(flag, None)

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Baseline:                {t1:.2f}s")
    print(f"XLA passes disabled:     {t2:.2f}s ({t1/t2:.2f}x)")


if __name__ == "__main__":
    main()
