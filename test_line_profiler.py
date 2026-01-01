"""Use line_profiler to find bottleneck lines in bitonic module during lowering."""

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


def kernel_simplified(
    logits_ref, k_vmem_ref, topk_vals_ref, topk_idxs_ref, valid_ref,
    max_depth_ref, cutoff_vals_ref, bins_topm_vals_ref, bins_topm_idxs_ref,
    termination_flag_ref, *, max_k, num_bins, bins_topm_unroll,
    bins_topm_schedule, replace_val,
):
    """Simplified kernel for profiling."""
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


def profile_with_line_profiler(shape=(16, 2048)):
    """Profile .lower() using line_profiler."""
    print("="*70)
    print("Line Profiler - Finding Bottleneck Lines")
    print("="*70)
    print(f"Shape: {shape}")
    print()

    try:
        from line_profiler import LineProfiler
    except ImportError:
        print("ERROR: line_profiler not installed")
        print("Install with: pip install line_profiler")
        return

    # Import all functions from bitonic module
    import tallax.tax.bitonic as bitonic_mod
    import tallax.tax.bitonic.topk as bitonic_topk_mod
    import tallax.tax.bitonic.sort as bitonic_sort_mod

    # Get all functions to profile
    functions_to_profile = []

    # From bitonic/__init__.py
    for name in dir(bitonic_mod):
        obj = getattr(bitonic_mod, name)
        if callable(obj) and not name.startswith('_') and hasattr(obj, '__code__'):
            functions_to_profile.append((f'bitonic.{name}', obj))

    # From bitonic/topk.py
    for name in dir(bitonic_topk_mod):
        obj = getattr(bitonic_topk_mod, name)
        if callable(obj) and not name.startswith('_') and hasattr(obj, '__code__'):
            functions_to_profile.append((f'bitonic.topk.{name}', obj))

    # From bitonic/sort.py
    for name in dir(bitonic_sort_mod):
        obj = getattr(bitonic_sort_mod, name)
        if callable(obj) and not name.startswith('_') and hasattr(obj, '__code__'):
            functions_to_profile.append((f'bitonic.sort.{name}', obj))

    print(f"Profiling {len(functions_to_profile)} functions from tallax.tax.bitonic:\n")
    for name, _ in functions_to_profile[:10]:
        print(f"  - {name}")
    if len(functions_to_profile) > 10:
        print(f"  ... and {len(functions_to_profile) - 10} more")
    print()

    # Create profiler
    lp = LineProfiler()
    for _, func in functions_to_profile:
        lp.add_function(func)

    # Create pallas function
    print("Creating pallas function...")
    pallas_fn = create_pallas_fn(shape)
    jitted = jax.jit(pallas_fn)

    # Profile the lowering
    print("Profiling jitted.lower()...")
    print("(This will take a while...)\n")

    t0 = time.perf_counter()
    lp.enable()
    lowered = jitted.lower()
    lp.disable()
    t1 = time.perf_counter()

    print(f"Lowering complete: {t1 - t0:.2f}s\n")
    print("="*70)
    print("LINE PROFILER RESULTS")
    print("="*70)

    # Print statistics
    lp.print_stats()

    # Also save to file
    output_file = f'/tmp/line_profiler_{shape[0]}_{shape[1]}.txt'
    with open(output_file, 'w') as f:
        lp.print_stats(stream=f)

    print(f"\nFull results saved to: {output_file}")


def main():
    # Profile (256, 2048) for full analysis
    profile_with_line_profiler(shape=(256, 2048))


if __name__ == "__main__":
    main()
