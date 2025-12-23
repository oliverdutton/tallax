# CSE Verification Results for Bitonic Sort - Final Report

## Executive Summary

This report presents the results of applying Common Subexpression Elimination (CSE) to the `bitonic_sort_arrays` function from tallax, testing with shapes (16, 1024) and (16, 32768).

### Key Findings

1. **CSE reaches fixpoint in 2 iterations** - No additional reduction beyond the first iteration
2. **Significant primitive count reduction** achieved for both test shapes
3. **Recursive counting through jit primitives** was critical - the original jaxpr contains many `jit` wrapper primitives
4. **Correctness verified** for (16, 1024) shape

---

## Test Configuration

- **Input Shapes**: (16, 1024) and (16, 32768)
- **Function**: `bitonic_sort_arrays` from `tallax._src.bitonic_topk`
- **Parameters**: `num_keys=1, axis=1, descending=False`
- **CSE Iterations**: Until fixpoint (max 10 iterations)

---

## Shape (16, 1024) Results

### Overall Reduction

| Metric | Count |
|--------|-------|
| **Original primitives** (with jit counted) | 12,449 |
| **Number of jit primitives** | 1,512 |
| **Original primitives** (recursing through jit) | 10,937 |
| **After 1 CSE iteration** | 7,732 (29.30% reduction) |
| **At fixpoint (2 iterations)** | 7,732 (29.30% reduction) |
| **Additional reduction at fixpoint** | 0 |

### Top Primitive Counts (Original)

```
convert_element_type: 1,700
select_n:             1,460
gt:                   1,441
add:                  1,284
xor:                    977
and:                    789
eq:                     704
```

### Most Significant Per-Primitive Reductions

| Primitive | Original | After CSE | Reduction | % Reduction |
|-----------|----------|-----------|-----------|-------------|
| **iota** | 170 | 2 | 168 | **98.82%** |
| **mul** | 104 | 2 | 102 | **98.08%** |
| **sign** | 104 | 2 | 102 | **98.08%** |
| **div** | 52 | 1 | 51 | **98.08%** |
| **rem** | 52 | 1 | 51 | **98.08%** |
| **sub** | 52 | 1 | 51 | **98.08%** |
| **and** | 789 | 169 | 620 | **78.58%** |
| **add** | 1,284 | 546 | 738 | **57.48%** |
| **gt** | 1,441 | 872 | 569 | **39.49%** |
| **convert_element_type** | 1,700 | 1,127 | 573 | **33.71%** |

### Verification

✅ **Verification PASSED**: Fixpoint jaxpr produces identical output to original
✅ **Correctness PASSED**: Output matches reference `jnp.sort()` implementation

---

## Shape (16, 32768) Results

### Overall Reduction

| Metric | Count |
|--------|-------|
| **Original primitives** (with jit counted) | 650,975 |
| **Number of jit primitives** | 86,250 |
| **Original primitives** (recursing through jit) | 564,725 |
| **After 1 CSE iteration** | 425,046 (24.73% reduction) |
| **At fixpoint (2 iterations)** | 425,046 (24.73% reduction) |
| **Additional reduction at fixpoint** | 0 |

### Top Primitive Counts (Original)

```
convert_element_type: 86,133
select_n:             86,133
gt:                   86,064
add:                  67,701
and:                  43,173
xor:                  43,056
eq:                   43,008
```

### Most Significant Per-Primitive Reductions

| Primitive | Original | After CSE | Reduction | % Reduction |
|-----------|----------|-----------|-----------|-------------|
| **iota** | 330 | 2 | 328 | **99.39%** |
| **mul** | 234 | 2 | 232 | **99.15%** |
| **sign** | 234 | 2 | 232 | **99.15%** |
| **div** | 117 | 1 | 116 | **99.15%** |
| **rem** | 117 | 1 | 116 | **99.15%** |
| **sub** | 117 | 1 | 116 | **99.15%** |
| **and** | 43,173 | 7,689 | 35,484 | **82.19%** |
| **add** | 67,701 | 25,090 | 42,611 | **62.94%** |
| **gt** | 86,064 | 50,696 | 35,368 | **41.09%** |
| **convert_element_type** | 86,133 | 61,447 | 24,686 | **28.66%** |

### Verification

⏱️ **Verification SKIPPED**: Evaluation takes too long for this large shape (jaxpr too complex)

---

## Key Insights

### 1. Importance of Recursing Through `jit` Primitives

The original jaxpr contains a large number of `jit` wrapper primitives:
- Shape (16, 1024): 1,512 jit primitives
- Shape (16, 32768): 86,250 jit primitives

**Critical Discovery**: CSE and primitive counting must recurse through these `jit` primitives to see the actual computation graph. When counting with `jit` as a primitive:
- (16, 1024): 12,449 total primitives
- When recursing through `jit`: 10,937 actual primitives

This is why the improved implementation properly handles jit primitives.

### 2. CSE Reaches Fixpoint Quickly

For both shapes, CSE reaches a fixpoint after just **2 iterations**:
- Iteration 1: Significant reduction (24-29%)
- Iteration 2: No additional reduction (fixpoint reached)

This indicates that the CSE algorithm is effective at finding all common subexpressions in a single pass, with the second iteration confirming convergence.

### 3. Arithmetic Operations Show Highest Reduction

Arithmetic and mathematical operations show the most dramatic reductions:
- `iota`, `mul`, `sign`, `div`, `rem`, `sub`: ~98-99% reduction
- These operations are computed once and reused extensively

### 4. Bitwise Operations Also Benefit Significantly

Bitwise operations show strong reductions:
- `and`: 78-82% reduction
- Indicates many redundant bitwise computations in the bitonic sort algorithm

### 5. Comparison and Selection Operations

Moderate but significant reductions:
- `add`: 57-63% reduction
- `gt`: 39-41% reduction
- These are core to the sorting algorithm but still have redundancy

### 6. Memory Operations Have Minimal Redundancy

Operations like `reshape`, `gather`, `slice`, `eq`, `lt` show little to no reduction, as expected - these typically operate on different data or represent unique operations.

### 7. Scaling Behavior

Comparing the two shapes:
- Larger shape (16, 32768) has 51.6× more primitives (564,725 vs 10,937)
- Similar reduction percentages (24.73% vs 29.30%)
- Even higher reduction rates for arithmetic ops in larger case (99.39% vs 98.82% for `iota`)

---

## Implementation Notes

### CSE Algorithm

The CSE implementation uses:
- **MD5 hashing** to identify duplicate computations based on:
  - Primitive type
  - Input variables
  - Parameters
- **Recursive application** to nested jaxprs (including through `jit` primitives)
- **Substitution map** to replace duplicate outputs with cached results

### Fixpoint Detection

Fixpoint is detected by comparing primitive counts between iterations:
- If counts are identical, no new eliminations were made
- Maximum 10 iterations (though only 2 were needed)

---

## Files

- `cse_bitonic_verify.py`: Original verification script
- `cse_bitonic_verify_improved.py`: Improved script with:
  - Recursive jit handling
  - Fixpoint iteration
  - Multiple shape testing
- `test_cse_jit.py`: Simple test to verify CSE behavior with jit primitives
- `cse_results.txt`: Raw output from improved verification

---

## Conclusion

CSE proves highly effective for optimizing the bitonic sort implementation:

1. **~25-29% overall primitive reduction** across different input sizes
2. **Near-elimination** of redundant arithmetic operations (98-99%)
3. **Significant reduction** in bitwise operations (78-82%)
4. **Fast convergence** to fixpoint (2 iterations)
5. **Correct semantics** preserved (verified for smaller shape)

The particularly high reduction rates for fundamental operations like `iota`, `mul`, `div`, and arithmetic operations suggest that the bitonic sort implementation could benefit significantly from CSE optimization in a production compiler.
