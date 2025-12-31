"""Test named scopes (without optimization barriers) on CPU lowering."""

import time
import jax
import jax.numpy as jnp
from functools import partial
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax.vllm.tpu_inference_sampling_as_standalone_file import TPUSupportedSamplingMetadata
from tallax.tax.divide_and_filter_topk.topk import binned_topk, bitonic_topk_arrays
from tallax.tax.utils import ceil_multiple, log2, to_32bit_dtype, get_dtype_info


def dynamic_topk_refs_with_named_scopes(
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
    """Kernel with named scopes (NO optimization barriers)."""
    # Initialize buffers
    block_token = logits_ref.shape[0]
    shape = (block_token, bins_topm_vals_ref.shape[1])

    pid = pl.program_id(0)
    token_slice = pl.dslice(pid * block_token, block_token)

    # NAMED SCOPE: Initialization
    with jax.named_scope("init_buffers"):
        bins_topm_vals_ref[token_slice] = jnp.full(
            shape, get_dtype_info(logits_ref).min, dtype=bins_topm_vals_ref.dtype
        )

        for i in range(block_token):
            max_depth_ref[pid * block_token + i] = max_k
        termination_flag_ref[0] = 0

    # Incremental binned top-k computation
    with jax.named_scope("incremental_binned_topk"):
        for completed_m, m in zip(bins_topm_schedule, bins_topm_schedule[1:]):

            @pl.when(termination_flag_ref[0] == 0)
            def _():
                # NAMED SCOPE: Each iteration
                with jax.named_scope(f"iteration_m{m}"):
                    # Compute binned top-m
                    with jax.named_scope("compute_binned_topk"):
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
                    with jax.named_scope("store_results"):
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
                    with jax.named_scope("convergence_check"):
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

    # Final top-k extraction
    global_topk_schedule = [max(x - 1, 0) for x in bins_topm_schedule[:-1]] + [
        bins_topm_schedule[-1]
    ]
    global_topk_schedule = tuple(sorted(set(bins_topm_schedule)))

    @pl.when(pl.program_id(0) == (pl.num_programs(0) - 1))
    def _():
        with jax.named_scope("final_topk_extraction"):
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
                    with jax.named_scope("bitonic_sort"):
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


def time_lowering_only(fn_callable, name="function"):
    """Time only the lowering stage."""
    print(f"\n{'='*60}")
    print(f"Lowering: {name}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    try:
        jitted = jax.jit(fn_callable)
        lowered = jitted.lower()
        t1 = time.perf_counter()
        lower_time = t1 - t0

        print(f"  ✓ Lowering successful: {lower_time:.2f}s")

        try:
            hlo_text = lowered.as_text()
            print(f"  HLO size: {len(hlo_text):,} characters")
        except:
            pass

        return {'time': lower_time, 'success': True}

    except Exception as e:
        t1 = time.perf_counter()
        lower_time = t1 - t0

        print(f"  ✗ Lowering failed: {lower_time:.2f}s")
        print(f"  Error: {type(e).__name__}: {str(e)[:200]}")

        return {'time': lower_time, 'success': False, 'error': str(e)}


def create_pallas_fn(logits, k, kernel_fn, max_k=128, block_token=8, num_bins=256,
                     bins_topm_schedule=(0, 5, 9), bins_topm_unroll=64):
    """Create pallas_call with specified kernel function."""
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


def test_lowering(shape, use_named_scopes=False, bins_topm_unroll=64, dtype=jnp.bfloat16, seed=42):
    """Test lowering time with or without named scopes."""
    num_tokens, vocab_size = shape

    print(f"\n{'#'*70}")
    print(f"Testing: shape={shape}, named_scopes={use_named_scopes}, unroll={bins_topm_unroll}")
    print(f"{'#'*70}")

    # Import the simple kernel
    from test_lowering_simple import dynamic_topk_refs_simple

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

    # Choose kernel
    kernel_fn = dynamic_topk_refs_with_named_scopes if use_named_scopes else dynamic_topk_refs_simple

    # Create the pallas function
    pallas_fn = create_pallas_fn(logits, k, kernel_fn, bins_topm_unroll=bins_topm_unroll)

    # Time lowering
    suffix = "named_scopes" if use_named_scopes else "baseline"
    result = time_lowering_only(pallas_fn, name=f"topk_{shape}_{suffix}")

    return result


def main():
    print("="*70)
    print("TPU Lowering Time Test - Named Scopes")
    print("="*70)
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print()
    print("NOTE: Optimization barriers are NOT supported in TPU Pallas lowering.")
    print("Testing only named scopes for HLO organization.")
    print()

    # Test shape
    shape = (16, 2048)

    # Test: Baseline vs Named Scopes
    print("\n" + "="*70)
    print("TEST: Baseline vs Named Scopes")
    print("="*70)

    jax.clear_caches()
    baseline = test_lowering(shape, use_named_scopes=False, bins_topm_unroll=64)

    jax.clear_caches()
    named_scopes = test_lowering(shape, use_named_scopes=True, bins_topm_unroll=64)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if baseline['success'] and named_scopes['success']:
        t_baseline = baseline['time']
        t_scopes = named_scopes['time']
        ratio = t_baseline / t_scopes

        print(f"\nNamed Scopes Impact:")
        print(f"  Baseline:           {t_baseline:.1f}s")
        print(f"  With named scopes:  {t_scopes:.1f}s")
        print(f"  Ratio:              {ratio:.2f}x")

        if ratio > 1.05:
            print("  ✓ Named scopes help!")
        elif ratio > 0.95:
            print("  → No significant change (expected for lowering)")
        else:
            print("  ✗ Named scopes hurt slightly")

        print("\nNOTE: Named scopes primarily help with:")
        print("  - HLO readability and debugging")
        print("  - Profiling and performance analysis")
        print("  - Not expected to significantly impact lowering time")


if __name__ == "__main__":
    main()
