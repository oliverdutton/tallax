# CPU Test Report for Tallax

## Test Execution Summary

### Tests that PASS on CPU (with interpret=True):

#### 1. **gather_test.py** ✅
- **37 test cases** (test_gather_correctness parametrized + test_gather_large_k_explicit)
- Tests: `tax.take_along_axis()` with various shapes and k values
- **Status**: All pass individually (~2.8s each, ~100s total)
- **Uses pallas_compatible**: Yes (`pallas_compatible_take_along_axis` in gather.py)
- **Issue**: Times out when run together (>120s), needs timeout mark

#### 2. **tumpy_test.py** ✅
- **16 test cases** (sort, argsort, take_along_axis tests)
- Tests: `tnp.sort()`, `tnp.argsort()`, `tnp.take_along_axis()`
- **Status**: Pass individually (~13s for sort, <6s for others)
- **Uses pallas_compatible**: Yes (via tax.take_along_axis, tax.sort)
- **Issue**: Times out when run together, needs timeout mark

#### 3. **bitonic_topk_test.py** ✅
- **test_bitonic_topk_axis1**: 6 test cases (shapes: (8,128), (16,256), (13,167) × 2 dtypes)
  - **Status**: All pass (~5-7s each)
  - **Uses pallas_compatible**: Yes (calls `pallas_compatible_bitonic_topk` directly on CPU)
  - **Checks work**: Yes, `verify_topk_output` validates correctness

- **test_top1_axis0_pallas**: 8 test cases (4 shapes × 2 dtypes)
  - **Status**: All pass (~4-6s each)
  - **Uses pallas_compatible**: Yes (`top1` function works with interpret=True)
  - **Checks work**: Yes, `verify_topk_output` validates correctness
  - **Note**: Requires dim0 to be power of 2

#### 4. **cumsum_test.py** ⚠️ **HAS BUGS**
- **40 test cases** (5 shapes × 2 axes × 2 dtypes × 2 reverse values)
- Tests: `tax.cumsum()` with various parameters
- **Status**:
  - ✅ **34 PASS**: All non-reverse cases + reverse on original 4 shapes
  - ❌ **6 FAIL**: reverse=True on shape (13,167) - integer overflow bug
- **Uses pallas_compatible**: Yes (`pallas_compatible_cumsum` in cumsum.py)
- **Checks work**: Yes for passing tests; **Bug in reverse implementation** for (13,167)
  - Integer overflow: actual values like `-2147475618` vs expected `8119`
  - Affects: axis 0 and 1, both int32 and float32
  - **Root cause**: `pallas_compatible_cumsum` reverse logic has bug with non-aligned shapes

### Tests that are SKIPPED on CPU:

#### 5. **sort_test.py**
- **108 test cases** (test_sort_comprehensive)
- **Status**: ALL SKIPPED
- **Reason**: `@pytest.mark.skipif(is_cpu_platform())`
- **Why**: "Sort tests require TPU/GPU - CPU uses interpret mode which is slow for comprehensive tests"
- **Are checks working**: N/A (intentionally skipped for performance)

#### 6. **top_k_test.py**
- **1 test case** (test_top_k)
- **Status**: SKIPPED
- **Reason**: `@pytest.mark.skipif(is_cpu_platform())`
- **Why**: "Top-k tests require TPU/GPU - CPU mode too slow"
- **Are checks working**: N/A (intentionally skipped)

### Tests that DON'T EXIST or are EXCLUDED:

#### 7. **fused_sampling_test.py**
- **Status**: Excluded from test runs (--ignore flag)
- **Reason**: ModuleNotFoundError for `tallax.tax.fused_sampling`

---

## Detailed Issues Found:

### ❌ **CRITICAL BUG: cumsum reverse with (13,167) shape**
```python
# Failing tests:
- test_cumsum_correctness[True-int32-0-shape4]  # axis=0, reverse=True
- test_cumsum_correctness[True-int32-1-shape4]  # axis=1, reverse=True
- test_cumsum_correctness[True-float32-0-shape4]
- test_cumsum_correctness[True-float32-1-shape4]
```

**Expected vs Actual**:
```
Expected: array([[8119, 8075, 8061, ..., 132, 118, 75], ...])
Actual:   array([[-2147475618, -2147475662, -2147475676, ..., -2147483605, ...]])
```

**Root cause**: The `pallas_compatible_cumsum` function in `tallax/tax/cumsum.py` has a bug in the reverse implementation when dealing with non-aligned shapes like (13, 167) that don't divide evenly into (NUM_SUBLANES, NUM_LANES) = (8, 128) tiles.

### ⚠️ **PERFORMANCE ISSUE: Tests timeout**

Many tests timeout when run together but pass individually:
- **gather_test.py**: 37 tests × ~3s = ~110s (close to timeout)
- **tumpy_test.py**: Some tests take 13s+

**Recommendation**: Add `@pytest.mark.timeout(180)` or similar to slow test files.

---

## Summary Statistics:

- **Total test cases examined**: ~175
- **Pass on CPU**: ~58 (bitonic_topk + tumpy + gather)
- **Fail on CPU**: 6 (cumsum reverse bug)
- **Skipped on CPU**: 109 (sort + top_k - intentional for performance)
- **Excluded**: 1 file (fused_sampling - missing module)

## Pallas Compatible Functions Status:

✅ **Working on CPU**:
- `pallas_compatible_take_along_axis` (gather.py)
- `pallas_compatible_bitonic_topk` (bitonic_topk.py)
- `top1` (bitonic_topk.py)
- `pallas_compatible_cumsum` (cumsum.py) - **mostly works, reverse has bug**

## Recommendations:

1. **Fix cumsum reverse bug** for non-aligned shapes
2. **Add pytest timeouts** to slow test files
3. **Consider skipping** large parametrize sets on CPU (e.g., only test subset of gather cases)
4. **Verify** cumsum reverse implementation handles padding correctly
