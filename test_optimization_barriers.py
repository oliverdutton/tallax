"""Test if optimization barriers reduce compilation time for top_bounded_k.

Strategy:
1. Add jax.lax.optimization_barrier between major kernel sections
2. Use jax.named_scope to structure the computation
3. Test if this reduces lowering/compilation time
"""

import time
import jax
import jax.numpy as jnp
import jax.lax as lax
from functools import partial
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax.vllm.tpu_inference_sampling_as_standalone_file import TPUSupportedSamplingMetadata
from tallax.tax.divide_and_filter_topk.topk import (
    binned_topk,
    _merge_unconverged_bins_topk,
    bitonic_topk_arrays,
    to_32bit_dtype,
    get_dtype_info,
)
from tallax.tax.utils import ceil_multiple, NUM_LANES, log2


def dynamic_topk_refs_with_barriers(
    logits_ref,
    k_smem_ref,
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
    guarantee_convergence: bool,
    replace_val: float | int | None,
):
    """Modified version with optimization barriers to reduce compilation time."""

    block_token = logits_ref.shape[0]
    shape = (block_token, bins_topm_vals_ref.shape[1])

    pid = pl.program_id(0)
    token_slice = pl.dslice(pid * block_token, block_token)

    # Initialize buffers
    with jax.named_scope("init_buffers"):
        bins_topm_vals_ref[token_slice] = jnp.full(
            shape, get_dtype_info(logits_ref).min, dtype=bins_topm_vals_ref.dtype
        )
        for i in range(block_token):
            max_depth_ref[pid * block_token + i] = max_k
        termination_flag_ref[0] = 0

    # BARRIER: Prevent fusion of initialization with main computation
    # This should help the compiler optimize init and main loop separately
    termination_flag_ref[0] = lax.optimization_barrier(termination_flag_ref[0])

    # Incremental binned top-k computation
    for completed_m, m in zip(bins_topm_schedule, bins_topm_schedule[1:]):

        @pl.when(termination_flag_ref[0] == 0)
        def _():
            # Wrap each major section in named_scope
            with jax.named_scope(f"binned_topk_m{m}"):
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

            # BARRIER: Separate binned_topk from result storage
            bins_topm_vals = [lax.optimization_barrier(v) for v in bins_topm_vals]
            bins_topm_idxs = [lax.optimization_barrier(idx) for idx in bins_topm_idxs]

            # Store results
            with jax.named_scope(f"store_results_m{m}"):
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

            # BARRIER: Separate storage from convergence checking
            pivot = lax.optimization_barrier(bins_topm_vals[m - 1].max(-1, keepdims=True))

            # Convergence check
            with jax.named_scope(f"convergence_check_m{m}"):
                num_larger = (
                    sum((v >= pivot) for v in bins_topm_vals[: m - 1])
                    .astype(to_32bit_dtype(logits_ref.dtype))
                    .sum(-1)
                )

                termination_flag_ref[0] = 0
                for i in range(block_token):
                    token_idx = pid * block_token + i
                    contains_topk = num_larger[i] >= k_smem_ref[token_idx]
                    termination_flag_ref[0] += contains_topk

                    current_max = max_depth_ref[token_idx]
                    max_depth_ref[token_idx] = jnp.where(
                        contains_topk & (current_max == max_k), m - 1, current_max
                    )
                    cutoff_vals_ref[token_idx] = pivot.squeeze(1)[i]

                @pl.when(termination_flag_ref[0] != block_token)
                def _():
                    termination_flag_ref[0] = 0

    # BARRIER: Separate incremental computation from bin packing
    termination_flag = lax.optimization_barrier(termination_flag_ref[0])

    # Bin packing optimization
    m_final = bins_topm_schedule[-1]
    if guarantee_convergence and (m_final < max_k):

        @pl.when(termination_flag == 0)
        def _():
            with jax.named_scope("bin_packing"):
                _merge_unconverged_bins_topk(
                    logits_ref,
                    bins_topm_vals_ref.at[token_slice],
                    bins_topm_idxs_ref.at[token_slice],
                    num_bins=num_bins,
                    m=m_final,
                    max_k=max_k,
                )

    global_topk_schedule = [max(x - 1, 0) for x in bins_topm_schedule[:-1]] + [
        bins_topm_schedule[-1] - (1 if guarantee_convergence else 0)
    ]
    global_topk_schedule = tuple(sorted(set(bins_topm_schedule)))

    # Final top-k extraction
    @pl.when(pl.program_id(0) == (pl.num_programs(0) - 1))
    def _():
        with jax.named_scope("final_topk_extraction"):
            global_max_depth = jnp.array(0)
            for i in range(max_depth_ref.shape[0]):
                global_max_depth = jnp.maximum(global_max_depth, max_depth_ref[i])

            # BARRIER: Separate max depth calculation from validation
            global_max_depth = lax.optimization_barrier(global_max_depth)

            valid_ref[0] = (
                (global_max_depth < bins_topm_schedule[-1])
                | (bins_topm_schedule[-1] >= max_k)
            ).astype(jnp.int32)

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
                    with jax.named_scope(f"bitonic_topk_depth{depth_upper}"):
                        vals_input = bins_topm_vals_ref[:, : depth_upper * num_bins]
                        idxs_input = bins_topm_idxs_ref[:, : depth_upper * num_bins]

                        # BARRIER: Separate input preparation from sorting
                        vals_input = lax.optimization_barrier(vals_input)
                        idxs_input = lax.optimization_barrier(idxs_input)

                        vals, idxs = bitonic_topk_arrays(
                            [vals_input, idxs_input],
                            num_keys=1,
                            k=max_k,
                        )
                        topk_vals_ref[...], topk_idxs_ref[...] = (
                            vals.astype(topk_vals_ref.dtype),
                            idxs,
                        )
                        if replace_val is not None:
                            idx = jax.lax.broadcasted_iota(jnp.int32, topk_vals_ref.shape, 1)
                            topk_vals_ref[...] = jnp.where(
                                idx < k_vmem_ref[...][:, None], topk_vals_ref[...], replace_val
                            )


def test_with_barriers(shape, dtype=jnp.bfloat16, seed=42):
    """Test compilation time with optimization barriers."""
    num_tokens, vocab_size = shape

    print(f"\n{'='*70}")
    print(f"Testing shape={shape} WITH optimization barriers")
    print(f"{'='*70}")

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

    # Parameters
    max_k = 128
    block_token = 8
    num_bins = 256
    bins_topm_schedule = (0, 5, 9)
    num_tokens_padded = ceil_multiple(num_tokens, block_token)

    # Buffer sizes
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

    # Time compilation
    pallas_fn = pl.pallas_call(
        partial(
            dynamic_topk_refs_with_barriers,
            max_k=max_k,
            num_bins=num_bins,
            bins_topm_unroll=64,
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
    )

    print("Starting compilation...")
    t0 = time.perf_counter()
    result = pallas_fn(logits, k, k)
    t1 = time.perf_counter()

    total_time = t1 - t0
    print(f"Total time (first call): {total_time:.2f}s")

    return total_time


def main():
    print("="*70)
    print("Testing Optimization Barriers for Compilation Time Reduction")
    print("="*70)
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")

    shapes = [(16, 2048), (256, 2048)]
    results = {}

    for shape in shapes:
        results[shape] = test_with_barriers(shape)

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    for shape in shapes:
        print(f"{shape}: {results[shape]:.2f}s")

    if len(shapes) == 2:
        ratio = results[shapes[1]] / results[shapes[0]]
        print(f"\nRatio (256,2048)/(16,2048): {ratio:.2f}x")

    print("\n" + "="*70)
    print("Compare with original results:")
    print("  Original (16, 2048): 16.49s")
    print("  Original (256, 2048): 254.36s")
    print("  Original ratio: 15.4x")
    print("="*70)


if __name__ == "__main__":
    main()
