"""Tests for i64_utils module with positive values and overflow cases."""

import jax.numpy as jnp
import numpy as np
from tallax.tax.i64_utils import i64_sum_dim1


def i32s_to_u64(i32s):
  """Convert list of i32 arrays [low, high, ...] to u64 values."""
  if len(i32s) == 0:
    return 0

  result = np.uint64(0)
  for i, i32 in enumerate(i32s):
    # Convert to numpy scalar first
    val = np.uint64(np.uint32(np.array(i32).item()))
    result += val << np.uint64(32 * i)
  return result


def test_simple_positive():
  """Test with simple positive integers."""
  print("\n=== Test: Simple Positive Integers ===")
  x = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.uint32)
  result = i64_sum_dim1(x, chunk_size=2)

  result_u64 = [i32s_to_u64([np.array(r[i, 0]) for r in result]) for i in range(x.shape[0])]
  expected = [10, 26]

  print(f"Input: {x}")
  print(f"Result i32s: {[r[:, 0] for r in result]}")
  print(f"Result u64: {result_u64}")
  print(f"Expected: {expected}")
  print("✓ PASS" if result_u64 == expected else "✗ FAIL")

  return result_u64 == expected


def test_large_positive_overflow():
  """Test with large positive integers that overflow i32 when summed."""
  print("\n=== Test: Large Positive with Overflow ===")
  large_val = 2**30  # 1,073,741,824
  x = jnp.array([[large_val] * 4], dtype=jnp.uint32)
  result = i64_sum_dim1(x, chunk_size=2)

  result_u64 = i32s_to_u64([np.array(r[0, 0]) for r in result])
  expected = large_val * 4  # 4,294,967,296

  print(f"Input: {large_val} repeated 4 times")
  print(f"Result i32s: {[r[0, 0] for r in result]}")
  print(f"Result u64: {result_u64}")
  print(f"Expected: {expected}")
  print("✓ PASS" if result_u64 == expected else "✗ FAIL")

  return result_u64 == expected


def test_harmonization_required():
  """Test case where harmonization of carries is required."""
  print("\n=== Test: Harmonization Required ===")
  val = 2**16 - 1  # 65535
  x = jnp.array([[val] * 256], dtype=jnp.uint32)
  result = i64_sum_dim1(x, chunk_size=128)

  result_u64 = i32s_to_u64([np.array(r[0, 0]) for r in result])
  expected = val * 256  # 16,776,960

  print(f"Input: {val} repeated 256 times")
  print(f"Result i32s: {[r[0, 0] for r in result]}")
  print(f"Result u64: {result_u64}")
  print(f"Expected: {expected}")
  print("✓ PASS" if result_u64 == expected else "✗ FAIL")

  return result_u64 == expected


def test_large_array_with_overflow():
  """Test with large array that causes significant overflow into high bits."""
  print("\n=== Test: Large Array with Overflow to High Bits ===")
  val = 2**31 - 1  # 2,147,483,647 (max signed i32, but treating as unsigned)
  n_values = 1000
  x = jnp.array([[val] * n_values], dtype=jnp.uint32)
  result = i64_sum_dim1(x, chunk_size=128)

  result_u64 = i32s_to_u64([np.array(r[0, 0]) for r in result])
  expected = int(val) * n_values  # 2,147,483,647,000

  print(f"Input: {val} repeated {n_values} times")
  print(f"Result i32s: {[r[0, 0] for r in result]}")
  print(f"Result u64: {result_u64}")
  print(f"Expected: {expected}")
  diff = abs(result_u64 - expected)
  print(f"Difference: {diff}")
  print("✓ PASS" if result_u64 == expected else "✗ FAIL")

  return result_u64 == expected


def test_zero_values():
  """Test with all zeros."""
  print("\n=== Test: All Zeros ===")
  x = jnp.zeros((2, 100), dtype=jnp.uint32)
  result = i64_sum_dim1(x, chunk_size=50)

  result_u64 = [i32s_to_u64([np.array(r[i, 0]) for r in result]) for i in range(x.shape[0])]
  expected = [0, 0]

  print(f"Input: zeros shape {x.shape}")
  print(f"Result i32s: {[r[:, 0] for r in result]}")
  print(f"Result u64: {result_u64}")
  print(f"Expected: {expected}")
  print("✓ PASS" if result_u64 == expected else "✗ FAIL")

  return result_u64 == expected


def test_multiple_rows():
  """Test with multiple rows to verify batching works correctly."""
  print("\n=== Test: Multiple Rows ===")
  n_rows = 5
  n_cols = 200
  x = jnp.arange(n_rows * n_cols, dtype=jnp.uint32).reshape(n_rows, n_cols)
  result = i64_sum_dim1(x, chunk_size=64)

  result_u64 = [i32s_to_u64([np.array(r[i, 0]) for r in result]) for i in range(n_rows)]
  expected = [int(x[i].sum()) for i in range(n_rows)]

  print(f"Input shape: {x.shape}")
  print(f"Result u64: {result_u64}")
  print(f"Expected: {expected}")
  print("✓ PASS" if result_u64 == expected else "✗ FAIL")

  return result_u64 == expected


def run_all_tests():
  """Run all tests and report results."""
  print("=" * 60)
  print("Running i64_sum_dim1 Tests (uint32 only)")
  print("=" * 60)

  tests = [
    ("Simple Positive", test_simple_positive),
    ("Large Positive Overflow", test_large_positive_overflow),
    ("Harmonization Required", test_harmonization_required),
    ("Large Array High Bits", test_large_array_with_overflow),
    ("All Zeros", test_zero_values),
    ("Multiple Rows", test_multiple_rows),
  ]

  results = []
  for name, test_fn in tests:
    try:
      passed = test_fn()
      results.append((name, passed))
    except Exception as e:
      print(f"\n✗ EXCEPTION in {name}: {e}")
      import traceback
      traceback.print_exc()
      results.append((name, False))

  print("\n" + "=" * 60)
  print("Test Summary")
  print("=" * 60)
  for name, passed in results:
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status}: {name}")

  total = len(results)
  passed = sum(1 for _, p in results if p)
  print(f"\nTotal: {passed}/{total} tests passed")
  print("=" * 60)


if __name__ == "__main__":
  run_all_tests()
