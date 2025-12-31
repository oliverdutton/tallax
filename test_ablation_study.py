"""Ablation study - systematically remove code sections to find bottleneck."""

import time
import jax
import jax.numpy as jnp
from functools import partial
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax.vllm.tpu_inference_sampling_as_standalone_file import TPUSupportedSamplingMetadata
from tallax.tax.divide_and_filter_topk.topk import binned_topk, bitonic_topk_arrays
from tallax.tax.utils import ceil_multiple, log2, to_32bit_dtype, get_dtype_info


# Version 1: Full kernel (baseline)
def kernel_full(
    logits_ref, k_vmem_ref, topk_vals_ref, topk_idxs_ref, valid_ref,
    max_depth_ref, cutoff_vals_ref, bins_topm_vals_ref, bins_topm_idxs_ref,
    termination_flag_ref, *, max_k, num_bins, bins_topm_unroll,
    bins_topm_schedule, replace_val,
):
    """Full kernel - baseline."""
    block_token = logits_ref.shape[0]
    shape = (block_token, bins_topm_vals_ref.shape[1])
    pid = pl.program_id(0)
    token_slice = pl.dslice(pid * block_token, block_token)

    # Init
    bins_topm_vals_ref[token_slice] = jnp.full(
        shape, get_dtype_info(logits_ref).min, dtype=bins_topm_vals_ref.dtype
    )
    for i in range(block_token):
        max_depth_ref[pid * block_token + i] = max_k
    termination_flag_ref[0] = 0

    # Main loop
    for completed_m, m in zip(bins_topm_schedule, bins_topm_schedule[1:]):
        @pl.when(termination_flag_ref[0] == 0)
        def _():
            bins_topm_vals, bins_topm_idxs = binned_topk(
                logits_ref, k=m,
                bins_topk_vals=[
                    bins_topm_vals_ref[token_slice, pl.dslice(i * num_bins, num_bins)].astype(to_32bit_dtype(logits_ref.dtype))
                    for i in range(m)
                ],
                bins_topk_idxs=[
                    bins_topm_idxs_ref[token_slice, pl.dslice(i * num_bins, num_bins)]
                    for i in range(m)
                ],
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

    # Final extraction
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


# Version 2: No final extraction (bitonic sort)
def kernel_no_final_sort(
    logits_ref, k_vmem_ref, topk_vals_ref, topk_idxs_ref, valid_ref,
    max_depth_ref, cutoff_vals_ref, bins_topm_vals_ref, bins_topm_idxs_ref,
    termination_flag_ref, *, max_k, num_bins, bins_topm_unroll,
    bins_topm_schedule, replace_val,
):
    """No final bitonic sort."""
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

    # Just write dummy output
    topk_vals_ref[...] = jnp.full_like(topk_vals_ref[...], get_dtype_info(topk_vals_ref).min)
    topk_idxs_ref[...] = jnp.zeros_like(topk_idxs_ref[...])
    valid_ref[0] = 1


# Version 3: No convergence checking
def kernel_no_convergence(
    logits_ref, k_vmem_ref, topk_vals_ref, topk_idxs_ref, valid_ref,
    max_depth_ref, cutoff_vals_ref, bins_topm_vals_ref, bins_topm_idxs_ref,
    termination_flag_ref, *, max_k, num_bins, bins_topm_unroll,
    bins_topm_schedule, replace_val,
):
    """No convergence checking."""
    block_token = logits_ref.shape[0]
    shape = (block_token, bins_topm_vals_ref.shape[1])
    pid = pl.program_id(0)
    token_slice = pl.dslice(pid * block_token, block_token)

    bins_topm_vals_ref[token_slice] = jnp.full(shape, get_dtype_info(logits_ref).min, dtype=bins_topm_vals_ref.dtype)
    for i in range(block_token):
        max_depth_ref[pid * block_token + i] = max_k
    termination_flag_ref[0] = 0

    for completed_m, m in zip(bins_topm_schedule, bins_topm_schedule[1:]):
        bins_topm_vals, bins_topm_idxs = binned_topk(
            logits_ref, k=m,
            bins_topk_vals=[bins_topm_vals_ref[token_slice, pl.dslice(i * num_bins, num_bins)].astype(to_32bit_dtype(logits_ref.dtype)) for i in range(m)],
            bins_topk_idxs=[bins_topm_idxs_ref[token_slice, pl.dslice(i * num_bins, num_bins)] for i in range(m)],
            num_bins=num_bins, completed_k=completed_m, unroll=bins_topm_unroll,
        )

        for i in range(completed_m, m):
            bins_topm_vals_ref[token_slice, pl.dslice(i * num_bins, num_bins)] = bins_topm_vals[i].astype(bins_topm_vals_ref.dtype)
            bins_topm_idxs_ref[token_slice, pl.dslice(i * num_bins, num_bins)] = bins_topm_idxs[i].astype(bins_topm_idxs_ref.dtype)

    # Dummy output
    topk_vals_ref[...] = jnp.full_like(topk_vals_ref[...], get_dtype_info(topk_vals_ref).min)
    topk_idxs_ref[...] = jnp.zeros_like(topk_idxs_ref[...])
    valid_ref[0] = 1


# Version 4: Minimal - just init
def kernel_minimal(
    logits_ref, k_vmem_ref, topk_vals_ref, topk_idxs_ref, valid_ref,
    max_depth_ref, cutoff_vals_ref, bins_topm_vals_ref, bins_topm_idxs_ref,
    termination_flag_ref, *, max_k, num_bins, bins_topm_unroll,
    bins_topm_schedule, replace_val,
):
    """Minimal - just initialization."""
    block_token = logits_ref.shape[0]
    pid = pl.program_id(0)
    token_slice = pl.dslice(pid * block_token, block_token)

    bins_topm_vals_ref[token_slice] = jnp.full(
        (block_token, bins_topm_vals_ref.shape[1]),
        get_dtype_info(logits_ref).min,
        dtype=bins_topm_vals_ref.dtype
    )
    topk_vals_ref[...] = jnp.full_like(topk_vals_ref[...], get_dtype_info(topk_vals_ref).min)
    topk_idxs_ref[...] = jnp.zeros_like(topk_idxs_ref[...])
    valid_ref[0] = 1


def time_lowering(kernel_fn, name, shape=(16, 2048)):
    """Time lowering for a kernel."""
    num_tokens, vocab_size = shape
    num_tokens_padded = ceil_multiple(num_tokens, 8)
    max_k = 128
    num_bins = 256
    bins_topm_schedule = (0, 5, 9)
    max_m = bins_topm_schedule[-1]
    buffer_size = max(max_m, 2 ** log2(max_m - 1)) * num_bins

    # Setup data
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

    print(f"\n{name:40s} ", end='', flush=True)
    t0 = time.perf_counter()
    jitted = jax.jit(pallas_fn)
    lowered = jitted.lower()
    t1 = time.perf_counter()
    print(f"{t1-t0:6.2f}s", flush=True)

    return t1 - t0


def main():
    print("="*70)
    print("Ablation Study - Remove Code Sections to Find Bottleneck")
    print("="*70)
    print(f"JAX version: {jax.__version__}")
    print(f"Shape: (16, 2048)\n")

    tests = [
        ("1. Full kernel (baseline)", kernel_full),
        ("2. No final bitonic sort", kernel_no_final_sort),
        ("3. No convergence checking", kernel_no_convergence),
        ("4. Minimal (init only)", kernel_minimal),
    ]

    results = {}
    for name, kernel_fn in tests:
        jax.clear_caches()
        results[name] = time_lowering(kernel_fn, name)

    print("\n" + "="*70)
    print("RESULTS - What Takes Time?")
    print("="*70)

    baseline = results["1. Full kernel (baseline)"]
    for name in results:
        t = results[name]
        reduction = baseline - t
        if name == "1. Full kernel (baseline)":
            print(f"{name:40s} {t:6.2f}s (baseline)")
        else:
            print(f"{name:40s} {t:6.2f}s (saved {reduction:5.2f}s, {reduction/baseline*100:4.1f}%)")

    print("\n" + "="*70)
    print("CONCLUSIONS")
    print("="*70)

    no_sort = results["2. No final bitonic sort"]
    no_conv = results["3. No convergence checking"]
    minimal = results["4. Minimal (init only)"]

    print(f"Time in final bitonic sort:     {baseline - no_sort:5.2f}s ({(baseline-no_sort)/baseline*100:4.1f}%)")
    print(f"Time in convergence checking:   {no_sort - no_conv:5.2f}s ({(no_sort-no_conv)/baseline*100:4.1f}%)")
    print(f"Time in binned_topk:            {no_conv - minimal:5.2f}s ({(no_conv-minimal)/baseline*100:4.1f}%)")
    print(f"Time in initialization:         {minimal:5.2f}s ({minimal/baseline*100:4.1f}%)")


if __name__ == "__main__":
    main()
