"""Investigate the final bitonic sort section in detail."""

import time
import jax
import jax.numpy as jnp
from functools import partial
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax.tax.divide_and_filter_topk.topk import binned_topk, bitonic_topk_arrays
from tallax.tax.utils import ceil_multiple, log2, to_32bit_dtype, get_dtype_info


# Version 1: Full final extraction (nested pl.when)
def kernel_full_final_extraction(
    logits_ref, k_vmem_ref, topk_vals_ref, topk_idxs_ref, valid_ref,
    max_depth_ref, cutoff_vals_ref, bins_topm_vals_ref, bins_topm_idxs_ref,
    termination_flag_ref, *, max_k, num_bins, bins_topm_unroll,
    bins_topm_schedule, replace_val,
):
    """Full final extraction with nested pl.when."""
    # Minimal init - just focus on final extraction
    topk_vals_ref[...] = jnp.full_like(topk_vals_ref[...], get_dtype_info(topk_vals_ref).min)
    topk_idxs_ref[...] = jnp.zeros_like(topk_idxs_ref[...])

    global_topk_schedule = tuple(sorted(set([max(x - 1, 0) for x in bins_topm_schedule[:-1]] + [bins_topm_schedule[-1]])))

    @pl.when(pl.program_id(0) == (pl.num_programs(0) - 1))
    def _():
        global_max_depth = jnp.array(0)
        for i in range(max_depth_ref.shape[0]):
            global_max_depth = jnp.maximum(global_max_depth, max_depth_ref[i])

        valid_ref[0] = ((global_max_depth < bins_topm_schedule[-1]) | (bins_topm_schedule[-1] >= max_k)).astype(jnp.int32)

        for depth_lower, depth_upper in zip(global_topk_schedule, global_topk_schedule[1:]):
            @pl.when(((global_max_depth > depth_lower) & (global_max_depth <= depth_upper)) |
                     ((depth_upper == global_topk_schedule[-1]) & (global_max_depth > depth_upper)))
            def _():
                vals_input = bins_topm_vals_ref[:, : depth_upper * num_bins]
                idxs_input = bins_topm_idxs_ref[:, : depth_upper * num_bins]
                vals, idxs = bitonic_topk_arrays([vals_input, idxs_input], num_keys=1, k=max_k)
                topk_vals_ref[...], topk_idxs_ref[...] = vals.astype(topk_vals_ref.dtype), idxs
                if replace_val is not None:
                    idx = jax.lax.broadcasted_iota(jnp.int32, topk_vals_ref.shape, 1)
                    topk_vals_ref[...] = jnp.where(idx < k_vmem_ref[...][:, None], topk_vals_ref[...], replace_val)


# Version 2: No nested pl.when (just do bitonic sort)
def kernel_simple_bitonic(
    logits_ref, k_vmem_ref, topk_vals_ref, topk_idxs_ref, valid_ref,
    max_depth_ref, cutoff_vals_ref, bins_topm_vals_ref, bins_topm_idxs_ref,
    termination_flag_ref, *, max_k, num_bins, bins_topm_unroll,
    bins_topm_schedule, replace_val,
):
    """Simplified - just do bitonic sort without nested conditions."""
    topk_vals_ref[...] = jnp.full_like(topk_vals_ref[...], get_dtype_info(topk_vals_ref).min)
    topk_idxs_ref[...] = jnp.zeros_like(topk_idxs_ref[...])

    @pl.when(pl.program_id(0) == (pl.num_programs(0) - 1))
    def _():
        # Just do the bitonic sort directly
        depth_upper = bins_topm_schedule[-1]
        vals_input = bins_topm_vals_ref[:, : depth_upper * num_bins]
        idxs_input = bins_topm_idxs_ref[:, : depth_upper * num_bins]
        vals, idxs = bitonic_topk_arrays([vals_input, idxs_input], num_keys=1, k=max_k)
        topk_vals_ref[...], topk_idxs_ref[...] = vals.astype(topk_vals_ref.dtype), idxs
        valid_ref[0] = 1


# Version 3: Bitonic sort called outside pl.when
def kernel_bitonic_outside_when(
    logits_ref, k_vmem_ref, topk_vals_ref, topk_idxs_ref, valid_ref,
    max_depth_ref, cutoff_vals_ref, bins_topm_vals_ref, bins_topm_idxs_ref,
    termination_flag_ref, *, max_k, num_bins, bins_topm_unroll,
    bins_topm_schedule, replace_val,
):
    """Call bitonic sort outside pl.when (on all programs)."""
    topk_vals_ref[...] = jnp.full_like(topk_vals_ref[...], get_dtype_info(topk_vals_ref).min)
    topk_idxs_ref[...] = jnp.zeros_like(topk_idxs_ref[...])

    # Bitonic sort called by ALL programs (not just last one)
    depth_upper = bins_topm_schedule[-1]
    vals_input = bins_topm_vals_ref[:, : depth_upper * num_bins]
    idxs_input = bins_topm_idxs_ref[:, : depth_upper * num_bins]
    vals, idxs = bitonic_topk_arrays([vals_input, idxs_input], num_keys=1, k=max_k)
    topk_vals_ref[...], topk_idxs_ref[...] = vals.astype(topk_vals_ref.dtype), idxs
    valid_ref[0] = 1


# Version 4: No bitonic sort at all
def kernel_no_bitonic(
    logits_ref, k_vmem_ref, topk_vals_ref, topk_idxs_ref, valid_ref,
    max_depth_ref, cutoff_vals_ref, bins_topm_vals_ref, bins_topm_idxs_ref,
    termination_flag_ref, *, max_k, num_bins, bins_topm_unroll,
    bins_topm_schedule, replace_val,
):
    """No bitonic sort - just dummy output."""
    topk_vals_ref[...] = jnp.full_like(topk_vals_ref[...], get_dtype_info(topk_vals_ref).min)
    topk_idxs_ref[...] = jnp.zeros_like(topk_idxs_ref[...])
    valid_ref[0] = 1


# Version 5: Only the loop over max_depth
def kernel_only_max_depth_loop(
    logits_ref, k_vmem_ref, topk_vals_ref, topk_idxs_ref, valid_ref,
    max_depth_ref, cutoff_vals_ref, bins_topm_vals_ref, bins_topm_idxs_ref,
    termination_flag_ref, *, max_k, num_bins, bins_topm_unroll,
    bins_topm_schedule, replace_val,
):
    """Only the max_depth loop."""
    topk_vals_ref[...] = jnp.full_like(topk_vals_ref[...], get_dtype_info(topk_vals_ref).min)
    topk_idxs_ref[...] = jnp.zeros_like(topk_idxs_ref[...])

    @pl.when(pl.program_id(0) == (pl.num_programs(0) - 1))
    def _():
        global_max_depth = jnp.array(0)
        for i in range(max_depth_ref.shape[0]):
            global_max_depth = jnp.maximum(global_max_depth, max_depth_ref[i])
        valid_ref[0] = global_max_depth.astype(jnp.int32)


def time_lowering(kernel_fn, name, shape=(16, 2048)):
    """Time lowering for a kernel."""
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
            partial(kernel_fn, max_k=max_k, num_bins=num_bins, bins_topm_unroll=64,
                    bins_topm_schedule=bins_topm_schedule, replace_val=-1e12),
            in_specs=(pl.BlockSpec((8, vocab_size), lambda i: (i, 0)), pl.BlockSpec(memory_space=pltpu.VMEM)),
            out_shape=output_shapes, scratch_shapes=tuple(scratch_shapes),
            grid=(pl.cdiv(num_tokens, 8),), out_specs=output_specs,
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=int(0.9 * 2**27)),
            interpret=False, backend='mosaic_tpu',
        )(logits, k)

    print(f"{name:50s} ", end='', flush=True)
    t0 = time.perf_counter()
    jitted = jax.jit(pallas_fn)
    lowered = jitted.lower()
    t1 = time.perf_counter()
    print(f"{t1-t0:6.2f}s", flush=True)

    return t1 - t0


def main():
    print("="*70)
    print("Bitonic Sort Section - Detailed Breakdown")
    print("="*70)
    print(f"JAX version: {jax.__version__}")
    print(f"Shape: (16, 2048)\n")

    tests = [
        ("1. Full final extraction (nested pl.when)", kernel_full_final_extraction),
        ("2. Simple bitonic (no nested when)", kernel_simple_bitonic),
        ("3. Bitonic outside pl.when", kernel_bitonic_outside_when),
        ("4. No bitonic sort", kernel_no_bitonic),
        ("5. Only max_depth loop", kernel_only_max_depth_loop),
    ]

    results = {}
    for name, kernel_fn in tests:
        jax.clear_caches()
        results[name] = time_lowering(kernel_fn, name)

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    baseline = results["1. Full final extraction (nested pl.when)"]
    for name in results:
        t = results[name]
        reduction = baseline - t
        if "Full" in name:
            print(f"{name:50s} {t:6.2f}s (baseline)")
        else:
            print(f"{name:50s} {t:6.2f}s (saved {reduction:5.2f}s, {reduction/baseline*100:4.1f}%)")

    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    simple = results["2. Simple bitonic (no nested when)"]
    outside = results["3. Bitonic outside pl.when"]
    no_bitonic = results["4. No bitonic sort"]
    only_loop = results["5. Only max_depth loop"]

    print(f"Time in nested pl.when structure:     {baseline - simple:5.2f}s ({(baseline-simple)/baseline*100:4.1f}%)")
    print(f"Time in pl.when(program_id check):    {simple - outside:5.2f}s ({(simple-outside)/baseline*100:4.1f}%)")
    print(f"Time in bitonic_topk_arrays:          {outside - no_bitonic:5.2f}s ({(outside-no_bitonic)/baseline*100:4.1f}%)")
    print(f"Time in max_depth loop:               {only_loop - no_bitonic:5.2f}s ({(only_loop-no_bitonic)/baseline*100:4.1f}%)")


if __name__ == "__main__":
    main()
