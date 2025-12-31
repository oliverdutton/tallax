"""Analyze why (256, 2048) compilation is so much slower than (16, 2048)."""

import jax
import jax.numpy as jnp
from tallax.tax.divide_and_filter_topk.topk import _top_bounded_k
from tallax.tax.utils import ceil_multiple

def analyze_grid_size(num_tokens, block_token=8):
    """Calculate the number of programs that will be launched."""
    num_tokens_padded = ceil_multiple(num_tokens, block_token)
    num_programs = num_tokens_padded // block_token
    return num_programs, num_tokens_padded

def analyze_buffer_sizes(max_k, num_bins, bins_topm_schedule):
    """Calculate buffer sizes used in the kernel."""
    from tallax.tax.utils import log2

    max_m = bins_topm_schedule[-1] if bins_topm_schedule else max_k
    buffer_size = max(max_m, 2 ** log2(max_m - 1)) * num_bins

    return {
        'max_m': max_m,
        'buffer_size': buffer_size,
        'buffer_size_elements': buffer_size,
    }

def main():
    print("=" * 70)
    print("COMPILATION ANALYSIS: (16, 2048) vs (256, 2048)")
    print("=" * 70)

    shapes = [(16, 2048), (256, 2048)]
    block_token = 8
    max_k = 128
    num_bins = 256
    bins_topm_schedule = (0, 5, 9)  # Note: (0,) gets prepended in the actual code

    for shape in shapes:
        num_tokens, vocab_size = shape

        print(f"\n{'='*70}")
        print(f"Shape: {shape}")
        print(f"{'='*70}")

        # Grid analysis
        num_programs, num_tokens_padded = analyze_grid_size(num_tokens, block_token)
        print(f"\nGrid configuration:")
        print(f"  block_token: {block_token}")
        print(f"  num_tokens: {num_tokens}")
        print(f"  num_tokens_padded: {num_tokens_padded}")
        print(f"  num_programs: {num_programs}")

        # Buffer analysis
        buffer_info = analyze_buffer_sizes(max_k, num_bins, bins_topm_schedule)
        print(f"\nBuffer sizes:")
        for key, value in buffer_info.items():
            print(f"  {key}: {value}")

        # Memory usage per program
        print(f"\nMemory per program (block_token={block_token}):")
        vmem_per_program = block_token * buffer_info['buffer_size_elements'] * 4  # float32 = 4 bytes
        print(f"  VMEM per program: {vmem_per_program / 1024 / 1024:.2f} MB")
        print(f"  Total VMEM across all programs: {vmem_per_program * num_programs / 1024 / 1024:.2f} MB")

        # Complexity analysis
        print(f"\nComplexity factors:")
        print(f"  Programs to compile: {num_programs}")
        print(f"  Elements per program: {block_token * vocab_size}")
        print(f"  Total elements: {num_tokens * vocab_size}")

    # Comparison
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")

    num_programs_16, _ = analyze_grid_size(16, block_token)
    num_programs_256, _ = analyze_grid_size(256, block_token)

    print(f"\nNum programs ratio (256,2048) / (16,2048):")
    print(f"  {num_programs_256} / {num_programs_16} = {num_programs_256 / num_programs_16:.1f}x")

    print(f"\nBatch size ratio:")
    print(f"  256 / 16 = {256 / 16:.1f}x")

    print(f"\nObserved compilation time ratio:")
    print(f"  157.39s / 9.36s = {157.39 / 9.36:.1f}x")

    print(f"\n{'='*70}")
    print("HYPOTHESIS")
    print(f"{'='*70}")
    print("""
The compilation slowdown is likely due to:

1. **Number of programs**: (256,2048) requires 32 programs vs 2 for (16,2048)
   - 16x more programs to compile

2. **Potential compilation complexity**: If compilation time scales
   superlinearly with the number of programs or with interactions between
   programs, this could explain the 16.7x slowdown vs the expected 16x.

3. **The divide-and-filter topk (top_bounded_k)** is the culprit, not
   the bitonic topk which only shows 4.3x slowdown.

To investigate further:
- Look at the HLO/StableHLO generated for both shapes
- Check if there are O(n^2) compilation patterns
- Profile the compiler itself
""")

if __name__ == "__main__":
    main()
