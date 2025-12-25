"""Line profiling script for bitonic_sort with interpret=False.

This script profiles various tallax functions in bitonic_sort using .lower()
to avoid execution (since we're on CPU but need TPU compilation).
"""

import jax
import jax.numpy as jnp
from line_profiler import LineProfiler

from tallax._src.bitonic_sort import (
    bitonic_sort,
    bitonic_sort_arrays,
    _bitonic_sort_substage,
    _compute_padded_shape,
    _compute_is_descending,
    _resplit,
    _rejoin,
)
from tallax._src.utils import (
    to_compressed_transpose_format,
    from_compressed_transpose_format,
    pad,
)
from tallax._src.sort import compare_and_swap


def main():
    """Profile bitonic_sort with shape (16, 8192) and stage_unroll=6."""

    # Create input data
    shape = (16, 8192)
    key = jax.random.PRNGKey(42)
    logits = jax.random.normal(key, shape, dtype=jnp.float32)

    print(f"Profiling bitonic_sort_arrays with shape {shape} and stage_unroll=6")
    print(f"interpret=False tracing mode (profiling compilation path)")
    print("-" * 80)

    # Set up line profiler
    profiler = LineProfiler()

    # Add functions to profile
    print("\nAdding functions to profile:")
    functions_to_profile = [
        # Main functions
        bitonic_sort_arrays,
        _bitonic_sort_substage.__wrapped__ if hasattr(_bitonic_sort_substage, '__wrapped__') else None,

        # Helper functions
        _compute_padded_shape,
        _compute_is_descending,
        _resplit,
        _rejoin,

        # Utils functions
        to_compressed_transpose_format,
        from_compressed_transpose_format,
        pad,
        compare_and_swap,
    ]

    for func in functions_to_profile:
        if func is not None:
            print(f"  - {func.__module__}.{func.__name__}")
            profiler.add_function(func)

    print("\n" + "=" * 80)
    print("Tracing bitonic_sort_arrays to generate compilation statistics...")
    print("=" * 80 + "\n")

    # Prepare the operands (pad them as the main function would)
    operands = [pad(logits, (8, 128), val='max')]

    # Wrap the execution to capture profiling
    profiler.enable()

    try:
        # Direct call to bitonic_sort_arrays to profile the actual execution path
        # This simulates what happens during compilation (interpret=False)
        # We use stage_unroll=None to avoid the transpose_refs requirement,
        # but we'll call _bitonic_sort_substage multiple times to profile that path
        print("Direct call to bitonic_sort_arrays (without stage_unroll)...")
        result = bitonic_sort_arrays(
            operands,
            num_keys=1,
            axis=1,
            descending=False,
            stage_unroll=None,  # Set to None to avoid transpose_refs requirement
            tile_unroll=None,
            transpose_refs=None
        )
        print(f"Result shape: {result[0].shape}")

        # Now let's manually trace through what stage_unroll=6 would do
        # by calling _bitonic_sort_substage directly
        print("\nManually profiling _bitonic_sort_substage calls (simulating stage_unroll=6)...")

        # Create test data for manual tracing
        from tallax._src.utils import NUM_SUBLANES, log2
        test_arr = pad(logits[:, :128], (16, 128), val='max')  # Smaller for testing
        test_tiles = to_compressed_transpose_format(test_arr)

        # Simulate first few stages with stage_unroll=6
        arrs_tiles = [test_tiles]
        batch_size = 16
        compression_length = test_tiles.shape[0]
        num_stages = log2(128)  # log2 of sort dimension

        print(f"Running stages 1-6 (stage_unroll=6) on smaller test data...")
        for stage in range(1, min(7, num_stages + 1)):  # stages 1-6
            for substage in range(stage)[::-1]:
                arrs_tiles = _bitonic_sort_substage(
                    arrs_tiles,
                    substage=substage,
                    stage=stage,
                    num_keys=1,
                    batch_size=batch_size,
                    sort_dim_offset=0,
                    compression_length=compression_length
                )
        print(f"Completed stage-unrolled substage profiling")

    except Exception as e:
        print(f"Error during profiling: {e}")
        import traceback
        traceback.print_exc()

    finally:
        profiler.disable()

    print("\n" + "=" * 80)
    print("LINE PROFILING RESULTS")
    print("=" * 80 + "\n")

    # Print profiling results
    profiler.print_stats()


if __name__ == "__main__":
    main()
