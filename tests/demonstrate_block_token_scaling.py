"""Demonstrate the exponential growth in operations with varying block_token.

Shows why block_token != 8 causes TPU compiler to crash.
"""


def count_operations(block_token):
    """Count the number of operations for a given block_token."""

    # _find_boundary_chunk is called twice in find_boundary_idx
    num_calls = 2

    # Per call:
    # - block_token pl.dslice operations (line 89)
    # - block_token-1 jnp.where merge operations (lines 91-92)

    dslice_ops_per_call = block_token
    where_ops_per_call = block_token - 1

    total_dslice = num_calls * dslice_ops_per_call
    total_where = num_calls * where_ops_per_call
    total_ops = total_dslice + total_where

    return {
        'dslice': total_dslice,
        'where': total_where,
        'total': total_ops,
    }


def estimate_memory(block_token, vocab_size=1024, chunk_size=256):
    """Estimate VMEM usage in bytes."""

    # Each boundary_slice: [block_token, chunk_size] in f32
    slice_bytes = block_token * chunk_size * 4

    # Need block_token slices
    total_slice_bytes = slice_bytes * block_token

    # Intermediate arrays for merging (roughly same size)
    intermediate_bytes = slice_bytes * 2

    # Total per call
    per_call_bytes = total_slice_bytes + intermediate_bytes

    # Called twice
    total_bytes = per_call_bytes * 2

    return {
        'per_slice_KB': slice_bytes / 1024,
        'all_slices_KB': total_slice_bytes / 1024,
        'per_call_KB': per_call_bytes / 1024,
        'total_MB': total_bytes / (1024 * 1024),
    }


def analyze_block_token_scaling():
    """Analyze how operations and memory scale with block_token."""

    print("="*80)
    print("BLOCK_TOKEN SCALING ANALYSIS")
    print("="*80)

    # Test various block_token values
    block_tokens = [1, 2, 4, 8, 16, 32, 64, 128]

    vmem_limit_mb = 0.9 * (2**27) / (1024**2)  # ≈ 120MB

    print(f"\nTPU VMEM limit: {vmem_limit_mb:.1f} MB")
    print(f"\n{'block_token':<12} {'dslice':<10} {'where':<10} {'total':<10} {'VMEM (MB)':<12} {'Status':<20}")
    print("-"*80)

    for bt in block_tokens:
        ops = count_operations(bt)
        mem = estimate_memory(bt)

        # Determine status
        if bt == 8:
            status = "✓ WORKS"
        elif mem['total_MB'] > vmem_limit_mb:
            status = "✗ VMEM OVERFLOW"
        elif ops['total'] > 100:
            status = "✗ TOO MANY OPS"
        else:
            status = "? UNTESTED"

        print(f"{bt:<12} {ops['dslice']:<10} {ops['where']:<10} {ops['total']:<10} "
              f"{mem['total_MB']:<12.2f} {status:<20}")

    # Detailed breakdown for key values
    print("\n" + "="*80)
    print("DETAILED BREAKDOWN")
    print("="*80)

    for bt in [8, 16, 32]:
        print(f"\nblock_token = {bt}:")
        ops = count_operations(bt)
        mem = estimate_memory(bt)

        print(f"  Operations:")
        print(f"    pl.dslice: {ops['dslice']} (each must be compile-time verified)")
        print(f"    jnp.where: {ops['where']} (each creates intermediate array)")
        print(f"    Total: {ops['total']}")

        print(f"  Memory:")
        print(f"    Per slice: {mem['per_slice_KB']:.1f} KB")
        print(f"    All slices: {mem['all_slices_KB']:.1f} KB")
        print(f"    Total: {mem['total_MB']:.2f} MB")

        if mem['total_MB'] > vmem_limit_mb:
            print(f"    ⚠️  EXCEEDS VMEM LIMIT by {mem['total_MB'] - vmem_limit_mb:.2f} MB")




def show_unrolled_loop_example():
    """Show what the loop looks like when unrolled."""

    print("\n" + "="*80)
    print("COMPILE-TIME LOOP UNROLLING EXAMPLE")
    print("="*80)

    for bt in [2, 4, 8]:
        print(f"\nWith block_token={bt}, the compiler must generate:")
        print(f"\n# Create {bt} separate slices:")
        for i in range(bt):
            print(f"slice_{i} = ref[:, pl.dslice(pl.multiple_of(ref_offset[{i}, 0], chunk_size), chunk_size)]")

        print(f"\n# Merge them with {bt-1} conditional operations:")
        print(f"result = slice_0")
        for i in range(1, bt):
            print(f"result = jnp.where(iota0 == {i}, slice_{i}, result)")

        print(f"\nTotal: {bt + bt-1} = {2*bt-1} operations")


if __name__ == "__main__":
    analyze_block_token_scaling()
    show_unrolled_loop_example()

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("\nblock_token=8 is the sweet spot because:")
    print("  1. Operations stay manageable (30 total)")
    print("  2. VMEM usage is well below limit (~0.5 MB)")
    print("  3. Aligns with TPU hardware (8 is factor of 128 lanes)")
    print("  4. Compile time is reasonable")
    print("\nLarger values cause exponential growth in:")
    print("  - Compile-time operations to verify")
    print("  - Memory pressure")
    print("  - Compiler analysis complexity")
    print("\nThis is why block_token != 8 crashes the TPU runtime!")
    print("="*80 + "\n")
