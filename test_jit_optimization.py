"""Test adding jax.jit with static_argnames to compare_and_swap and bitonic_sort_substage."""

import time
import jax
import jax.numpy as jnp
from functools import partial
from jax.experimental import pallas as pl

from tallax.tax.divide_and_filter_topk.topk import binned_topk, bitonic_topk_arrays
from tallax.tax.utils import ceil_multiple, log2, to_32bit_dtype, get_dtype_info


print(f"JAX version: {jax.__version__}\n")


def test_simplified_kernel(
    logits_ref, k_vmem_ref, topk_vals_ref, topk_idxs_ref, valid_ref,
    max_depth_ref, cutoff_vals_ref, bins_topm_vals_ref, bins_topm_idxs_ref,
    termination_flag_ref, *, max_k, num_bins, bins_topm_unroll,
    bins_topm_schedule, replace_val,
):
    """Simplified kernel for testing - no nested pl.when."""
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

    # Simplified final extraction (no nested pl.when)
    @pl.when(pl.program_id(0) == (pl.num_programs(0) - 1))
    def _():
        depth_upper = bins_topm_schedule[-1]
        vals_input = bins_topm_vals_ref[:, : depth_upper * num_bins]
        idxs_input = bins_topm_idxs_ref[:, : depth_upper * num_bins]
        vals, idxs = bitonic_topk_arrays([vals_input, idxs_input], num_keys=1, k=max_k)
        topk_vals_ref[...], topk_idxs_ref[...] = vals.astype(topk_vals_ref.dtype), idxs
        valid_ref[0] = 1


def apply_jit_patches():
    """Apply jax.jit to compare_and_swap and bitonic_sort_substage."""
    import tallax.tax.bitonic.sort as sort_module

    # Store original functions
    if not hasattr(sort_module, '_original_compare_and_swap'):
        sort_module._original_compare_and_swap = sort_module.compare_and_swap
        sort_module._original_bitonic_sort_substage = sort_module.bitonic_sort_substage

    # Create jitted versions
    @partial(jax.jit, static_argnames=('num_keys', 'has_unique_key'))
    def compare_and_swap_jitted(lefts, rights, *, num_keys, is_descending,
                                 is_right_half=None, has_unique_key=False):
        return sort_module._original_compare_and_swap(
            lefts, rights, num_keys=num_keys, is_descending=is_descending,
            is_right_half=is_right_half, has_unique_key=has_unique_key
        )

    @partial(jax.jit, static_argnames=(
        'substage', 'num_keys', 'batch_size', 'sort_dim_offset',
        'full_size', 'concat_threshold', 'max_reduce'
    ))
    def bitonic_sort_substage_jitted(arrs_tiles, *, substage, num_keys, batch_size,
                                      stage=None, sort_dim_offset=0, full_size=None,
                                      concat_threshold=None, max_reduce=False):
        return sort_module._original_bitonic_sort_substage(
            arrs_tiles, substage=substage, num_keys=num_keys, batch_size=batch_size,
            stage=stage, sort_dim_offset=sort_dim_offset, full_size=full_size,
            concat_threshold=concat_threshold, max_reduce=max_reduce
        )

    # Monkey patch
    sort_module.compare_and_swap = compare_and_swap_jitted
    sort_module.bitonic_sort_substage = bitonic_sort_substage_jitted


def remove_jit_patches():
    """Remove jax.jit patches and restore original functions."""
    import tallax.tax.bitonic.sort as sort_module

    if hasattr(sort_module, '_original_compare_and_swap'):
        sort_module.compare_and_swap = sort_module._original_compare_and_swap
        sort_module.bitonic_sort_substage = sort_module._original_bitonic_sort_substage


def test_with_and_without_jit(shape, use_jit):
    """Test lowering time with or without jit optimization."""
    batch_size, seq_len = shape
    max_k = 128
    num_bins = 9
    bins_topm_unroll = True

    # Calculate schedule
    bins_topm_schedule = [1]
    current = 1
    while current < max_k:
        current = min(current * 2, ceil_multiple(max_k, num_bins) // num_bins)
        bins_topm_schedule.append(current)
    bins_topm_schedule = tuple(bins_topm_schedule)

    max_topm = bins_topm_schedule[-1] * num_bins

    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Max k: {max_k}")
    print(f"Num bins: {num_bins}")
    print(f"Max topm: {max_topm}")
    print()

    # Create test data
    logits = jnp.ones((batch_size, seq_len), dtype=jnp.bfloat16)
    k = jnp.full((batch_size,), max_k, dtype=jnp.int32)

    # Grid and block specs
    block_token = 8
    grid = (batch_size // block_token,)

    logits_spec = pl.BlockSpec((block_token, seq_len), lambda i: (i * block_token, 0))
    k_spec = pl.BlockSpec((batch_size,), lambda i: (0,))
    topk_vals_spec = pl.BlockSpec((batch_size, max_k), lambda i: (0, 0))
    topk_idxs_spec = pl.BlockSpec((batch_size, max_k), lambda i: (0, 0))
    valid_spec = pl.BlockSpec((1,), lambda i: (0,))

    # Scratch space specs
    bins_topm_vals_scratch = pl.BlockSpec((batch_size, max_topm), lambda i: (0, 0))
    bins_topm_idxs_scratch = pl.BlockSpec((batch_size, max_topm), lambda i: (0, 0))
    max_depth_scratch = pl.BlockSpec((batch_size,), lambda i: (0,))
    cutoff_vals_scratch = pl.BlockSpec((batch_size,), lambda i: (0,))
    termination_flag_scratch = pl.BlockSpec((1,), lambda i: (0,))

    # Apply or remove jit patches
    if use_jit:
        print("✓ Applying @jax.jit to compare_and_swap and bitonic_sort_substage")
        apply_jit_patches()
    else:
        print("✗ Using original functions (no jit)")
        remove_jit_patches()
    print()

    # Create pallas call
    t0 = time.perf_counter()

    topk_pallas = pl.pallas_call(
        partial(
            test_simplified_kernel,
            max_k=max_k,
            num_bins=num_bins,
            bins_topm_unroll=bins_topm_unroll,
            bins_topm_schedule=bins_topm_schedule,
            replace_val=None,
        ),
        grid=grid,
        out_shape=[
            jax.ShapeDtypeStruct((batch_size, max_k), jnp.bfloat16),
            jax.ShapeDtypeStruct((batch_size, max_k), jnp.int32),
            jax.ShapeDtypeStruct((1,), jnp.int32),
        ],
        in_specs=[logits_spec, k_spec],
        out_specs=[topk_vals_spec, topk_idxs_spec, valid_spec],
        scratch_shapes=[
            bins_topm_vals_scratch,
            bins_topm_idxs_scratch,
            max_depth_scratch,
            cutoff_vals_scratch,
            termination_flag_scratch,
        ],
        compiler_params=dict(mosaic=dict(dimension_semantics=("parallel", "arbitrary"))),
        interpret=False,
        backend='mosaic_tpu',
    )

    t1 = time.perf_counter()
    print(f"[+{t1-t0:.2f}s] Pallas call created\n")

    # JIT
    t_jit_start = time.perf_counter()
    jitted = jax.jit(topk_pallas)
    t_jit_end = time.perf_counter()
    jit_time = t_jit_end - t_jit_start
    print(f"[+{t_jit_end-t0:.2f}s] jax.jit() completed ({jit_time*1000:.1f} ms)\n")

    # Lower
    print("Starting jitted.lower()...\n")
    t_lower_start = time.perf_counter()
    lowered = jitted.lower(logits, k)
    t_lower_end = time.perf_counter()
    lower_time = t_lower_end - t_lower_start
    print(f"[+{t_lower_end-t0:.2f}s] jitted.lower() COMPLETE ({lower_time:.2f}s)\n")

    # HLO extraction
    t_hlo_start = time.perf_counter()
    hlo = lowered.as_text()
    t_hlo_end = time.perf_counter()
    hlo_time = t_hlo_end - t_hlo_start

    print(f"HLO size: {len(hlo):,} chars")
    print(f"HLO extraction: {hlo_time*1000:.1f} ms")
    print()

    return {
        'lower_time': lower_time,
        'hlo_size': len(hlo),
        'hlo_time': hlo_time,
    }


def main():
    print("="*70)
    print("Testing @jax.jit Optimization on Bitonic Functions")
    print("="*70)
    print()

    shape = (16, 2048)

    # Test without jit (baseline)
    print("="*70)
    print("Test 1: WITHOUT JIT (baseline)")
    print("="*70)
    print()
    results_baseline = test_with_and_without_jit(shape, use_jit=False)

    print()
    print("="*70)
    print("BASELINE RESULTS")
    print("="*70)
    print(f"  Lowering time:    {results_baseline['lower_time']:.2f}s")
    print(f"  HLO size:         {results_baseline['hlo_size']:,} chars")
    print("="*70)
    print()
    print()

    # Test with jit
    print("="*70)
    print("Test 2: WITH JIT")
    print("="*70)
    print()
    results_jit = test_with_and_without_jit(shape, use_jit=True)

    print()
    print("="*70)
    print("WITH JIT RESULTS")
    print("="*70)
    print(f"  Lowering time:    {results_jit['lower_time']:.2f}s")
    print(f"  HLO size:         {results_jit['hlo_size']:,} chars")
    print("="*70)
    print()
    print()

    # Comparison
    print("="*70)
    print(f"FINAL COMPARISON - {shape}")
    print("="*70)
    print()

    baseline = results_baseline['lower_time']
    optimized = results_jit['lower_time']

    print(f"Baseline (no jit):        {baseline:8.2f}s")
    print(f"With jit optimization:    {optimized:8.2f}s")
    print()

    if optimized < baseline:
        speedup = baseline / optimized
        improvement = (baseline - optimized) / baseline * 100
        print(f"✓ SPEEDUP: {speedup:.2f}x ({improvement:.1f}% faster)")
        print(f"  Time saved: {baseline - optimized:.2f}s")
    elif optimized > baseline:
        slowdown = optimized / baseline
        regression = (optimized - baseline) / baseline * 100
        print(f"✗ SLOWDOWN: {slowdown:.2f}x ({regression:.1f}% slower)")
        print(f"  Time lost: {optimized - baseline:.2f}s")
    else:
        print("= NO CHANGE")

    print()
    print("="*70)


if __name__ == "__main__":
    main()
