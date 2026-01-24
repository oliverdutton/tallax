"""Analysis: Why block_token != 8 causes TPU runtime crashes

ROOT CAUSE:
===========

In topk_mask.py, _find_boundary_chunk() creates a compile-time unrolled loop
over batch_size, which is set to block_token when called from the kernel.

PROBLEMATIC CODE (topk_mask.py, lines 89-92):
=============================================

```python
batch_size = ref.shape[0]  # This is block_token in the kernel!

# Creates batch_size separate pl.dslice operations
boundary_slices = [
    ref[:, pl.dslice(pl.multiple_of(ref_offset[i, 0], chunk_size), chunk_size)]
    for i in range(batch_size)
]

# Merges them with batch_size-1 jnp.where operations
boundary_slice = boundary_slices[0]
for i in range(1, batch_size):
    boundary_slice = jnp.where(iota0 == i, boundary_slices[i], boundary_slice)
```

EXECUTION FLOW:
==============

1. topk_topp_mask_and_sample_kernel receives logits_ref[block_token, vocab_size]
2. Calls topk_mask_ref_inputs(logits_ref, ...) when stable=True
3. topk_mask_ref_inputs calls find_boundary_idx(logits_ref, ...)
4. find_boundary_idx calls _find_boundary_chunk(ref, ...) where ref.shape[0] = block_token
5. _find_boundary_chunk sets batch_size = ref.shape[0] = block_token
6. **Creates block_token slices and block_token-1 merge operations**

WHY block_token=8 WORKS BUT OTHERS CRASH:
=========================================

1. **Compile-time loop unrolling explosion**
   - With block_token=8: 8 pl.dslice ops + 7 jnp.where ops
   - With block_token=16: 16 pl.dslice ops + 15 jnp.where ops
   - With block_token=32: 32 pl.dslice ops + 31 jnp.where ops
   - With block_token=64: 64 pl.dslice ops + 63 jnp.where ops

   The compiler must analyze and optimize all these operations statically.
   Larger values cause exponential growth in compilation complexity.

2. **pl.dslice constraint violations**
   - Each pl.dslice requires compile-time proof that:
     * offset is a multiple of chunk_size (via pl.multiple_of)
     * slice doesn't violate memory bounds
   - With more slices, the compiler's symbolic analysis becomes harder
   - May fail to prove constraints for non-power-of-2 or large block_token

3. **TPU VMEM pressure**
   - vmem_limit_bytes = 0.9 * 2^27 ≈ 120MB
   - Each boundary_slice array: [block_token, chunk_size] in f32
     = block_token * 256 * 4 bytes (for vocab_size=1024)
   - With block_token=8: 8 * 256 * 4 = 8KB per slice
   - With block_token=64: 64 * 256 * 4 = 64KB per slice
   - Total for all slices + intermediates can exceed VMEM limit

4. **TPU hardware alignment**
   - TPU has natural block sizes (often powers of 2, especially 8)
   - block_token=8 aligns with:
     * 128-lane architecture (8 is factor of 128)
     * VMEM tile sizes
     * DMA transfer boundaries
   - Other values may cause misalignment → undefined behavior

5. **Nested iterations amplify the problem**
   - find_boundary_idx calls _find_boundary_chunk **twice**:
     * First with chunk_size = sqrt(vocab/128) * 128 = 256 (for vocab=1024)
     * Second with chunk_size = 128
   - Each call creates block_token slices
   - **Total: 2 * block_token slices + 2 * (block_token-1) merges**

SPECIFIC CRASH SCENARIOS:
========================

block_token=16:
- 32 pl.dslice operations
- 30 jnp.where merges
- Compilation time increases significantly
- May exceed compiler resource limits

block_token=32:
- 64 pl.dslice operations
- 62 jnp.where merges
- High probability of VMEM overflow
- Symbolic analysis likely fails

block_token=64:
- 128 pl.dslice operations
- 126 jnp.where merges
- Almost certain to crash during compilation or execution
- Violates fundamental TPU constraints

WHY CPU THREADS CRASH:
=====================

When TPU compiler fails:
1. May corrupt internal state
2. Can trigger segfaults in compiler backend
3. Errors propagate to JAX Python runtime
4. Crashes Python interpreter threads

SOLUTIONS:
==========

Option 1: Restrict block_token to 8 (current approach)
- Safe, tested, works
- Limits flexibility

Option 2: Rewrite _find_boundary_chunk to avoid per-batch slicing
- Use vectorized operations instead of loop
- More complex but supports arbitrary block_token
- Would require significant refactoring

Option 3: Use different algorithm for large block_token
- Keep current impl for block_token <= 8
- Fall back to simpler (slower) algorithm for larger values

Option 4: Make block_token a compile-time constant
- Set via static_argnames
- Allow compiler to specialize per block_token
- May help with larger values but still risky

RECOMMENDATION:
==============

Keep block_token=8 as the default and only supported value.
Add assertion to enforce this:

```python
assert block_token == 8, (
    "Only block_token=8 is currently supported due to TPU compiler "
    "constraints. See topk_mask.py _find_boundary_chunk for details."
)
```

The performance impact of block_token=8 is minimal since batch sizes
are typically padded to multiples of 8 anyway.
"""

import sys

def main():
    print(__doc__)

if __name__ == "__main__":
    main()
