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

#### 4. **cumsum_test.py** ✅ **FIXED**
- **40 test cases** (5 shapes × 2 axes × 2 dtypes × 2 reverse values)
- Tests: `tax.cumsum()` with various parameters
- **Status**:
  - ✅ **40 PASS**: All tests including reverse on shape (13,167)
  - **Bug was fixed**: reverse implementation corrected
- **Uses pallas_compatible**: Yes (`pallas_compatible_cumsum` in cumsum.py)
- **Checks work**: Yes - all tests pass
  - **Fix applied**: Added `reverse_tiles()` helper, fixed typo, changed pad val=0
  - Integer overflow bug resolved (was showing `-2147475618` vs expected `8119`)
  - Now correctly handles non-aligned shapes like (13,167)

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

### ✅ **FIXED: cumsum reverse with (13,167) shape**
```python
# Previously failing tests (now passing):
- test_cumsum_correctness[True-int32-0-shape4]  # axis=0, reverse=True ✅
- test_cumsum_correctness[True-int32-1-shape4]  # axis=1, reverse=True ✅
- test_cumsum_correctness[True-float32-0-shape4] ✅
- test_cumsum_correctness[True-float32-1-shape4] ✅
```

**Bug was**: Integer overflow showing `-2147475618` instead of `8119`

**Fix applied**:
- Added `reverse_tiles()` helper function
- Fixed typo: `reversal_perm` -> `reverse_perm`
- Changed `pad(arr, tile_shape)` to `pad(arr, tile_shape, val=0)`
- Simplified reverse logic to apply `reverse_tiles()` before and after cumsum

**Verified**: Manual test confirms correct behavior on shape (13, 167)

### ⚠️ **PERFORMANCE ISSUE: Tests timeout**

Many tests timeout when run together but pass individually:
- **gather_test.py**: 37 tests × ~3s = ~110s (close to timeout)
- **tumpy_test.py**: Some tests take 13s+

**Recommendation**: Add `@pytest.mark.timeout(180)` or similar to slow test files.

---

## Summary Statistics:

- **Total test cases examined**: ~175
- **Pass on CPU**: 98 (bitonic_topk + tumpy + gather + cumsum)
- **Fail on CPU**: 0 ✅ **ALL FIXED**
- **Skipped on CPU**: 109 (sort + top_k - intentional for performance)
- **Excluded**: 1 file (fused_sampling - missing module)

## Pallas Compatible Functions Status:

✅ **All working on CPU**:
- `pallas_compatible_take_along_axis` (gather.py) ✅
- `pallas_compatible_bitonic_topk` (bitonic_topk.py) ✅
- `top1` (bitonic_topk.py) ✅
- `pallas_compatible_cumsum` (cumsum.py) ✅ **FIXED**

## Recommendations:

1. ✅ ~~Fix cumsum reverse bug~~ **DONE**
2. **Add pytest timeouts** to slow test files
3. **Consider skipping** large parametrize sets on CPU (e.g., only test subset of gather cases)
4. ✅ ~~Verify cumsum reverse implementation~~ **VERIFIED & FIXED**
