# CSE (Common Subexpression Elimination) Results for Bitonic Sort

## Test Configuration
- **Input Shape**: (16, 1024)
- **Function**: `bitonic_sort_arrays` from `tallax._src.bitonic_topk`
- **Parameters**: `num_keys=1, axis=1, descending=False`

## Overall Results

### Primitive Count Reduction
- **Original Total Primitives**: 12,449
- **CSE'd Total Primitives**: 9,142
- **Reduction**: 3,307 primitives (26.56%)

## Per-Primitive Reductions

### Most Significant Reductions
1. **iota**: 170 → 2 (98.82% reduction)
2. **mul**: 104 → 2 (98.08% reduction)
3. **sign**: 104 → 2 (98.08% reduction)
4. **div**: 52 → 1 (98.08% reduction)
5. **rem**: 52 → 1 (98.08% reduction)
6. **sub**: 52 → 1 (98.08% reduction)
7. **and**: 789 → 169 (78.58% reduction)
8. **add**: 1,284 → 546 (57.48% reduction)
9. **gt**: 1,441 → 872 (39.49% reduction)
10. **convert_element_type**: 1,700 → 1,127 (33.71% reduction)

### Other Notable Reductions
- **ne**: 440 → 338 (23.18% reduction)
- **jit**: 1,512 → 1,410 (6.75% reduction)
- **select_n**: 1,460 → 1,409 (3.49% reduction)
- **xor**: 977 → 950 (2.76% reduction)

### Primitives with No Reduction
- **eq**: 704 (no change)
- **lt**: 528 (no change)
- **reshape**: 528 (no change)
- **gather**: 528 (no change)
- **slice**: 16 (no change)
- **split**: 3 (no change)

## Verification

✅ **Verification PASSED**: CSE'd jaxpr produces identical output to original
✅ **Correctness PASSED**: Output matches reference `jnp.sort()` implementation

## Key Insights

1. **Arithmetic Operations**: The CSE transformation was particularly effective at eliminating redundant arithmetic operations (`iota`, `mul`, `sign`, `div`, `rem`, `sub`), achieving ~98% reduction for most of these.

2. **Bitwise Operations**: Significant reduction in `and` operations (78.58%), indicating many redundant bitwise computations in the original implementation.

3. **Comparison Operations**: Good reduction in comparison operations like `add` (57.48%) and `gt` (39.49%).

4. **Type Conversions**: Reduced `convert_element_type` by 33.71%, eliminating redundant type conversions.

5. **Control Flow**: Minor reductions in control flow primitives (`select_n`, `jit`), suggesting these are mostly unique operations.

6. **Memory Operations**: No reduction in memory operations (`reshape`, `gather`, `slice`), as these typically operate on different data or are inherently unique.

## Implementation

The CSE implementation uses MD5 hashing to identify duplicate computations based on:
- Primitive type
- Input variables
- Parameters

It recursively applies CSE to nested jaxprs and maintains a substitution map to replace duplicate computation outputs with cached results.

## Files

- `cse_bitonic_verify.py`: Verification script that applies CSE and reports primitive counts
- Implementation uses `jax.core.eval_jaxpr` for jaxpr evaluation
