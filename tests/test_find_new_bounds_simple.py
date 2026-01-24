"""Simple test for find_new_bounds_unrolled."""

import jax.numpy as jnp
from tallax.vllm.binary_search import find_new_bounds_unrolled


def test_case_1():
    """Test case: [FALSE, FALSE, TRUE]"""
    pivots = [jnp.array([[1.0]]), jnp.array([[2.0]]), jnp.array([[3.0]])]
    predicates = [jnp.array([[False]]), jnp.array([[False]]), jnp.array([[True]])]

    new_l, new_r, any_false, any_true = find_new_bounds_unrolled(pivots, predicates)

    print("Test 1: predicates = [FALSE, FALSE, TRUE]")
    print(f"  pivots = [1.0, 2.0, 3.0]")
    print(f"  Expected: new_l=2.0 (last FALSE), new_r=3.0 (first TRUE)")
    print(f"  Got: new_l={float(new_l[0,0])}, new_r={float(new_r[0,0])}")
    print(f"  any_false={bool(any_false)}, any_true={bool(any_true)}")

    assert float(new_l[0,0]) == 2.0, f"Expected new_l=2.0, got {float(new_l[0,0])}"
    assert float(new_r[0,0]) == 3.0, f"Expected new_r=3.0, got {float(new_r[0,0])}"
    assert bool(any_false) == True
    assert bool(any_true) == True
    print("  ✓ PASSED\n")


def test_case_2():
    """Test case: [FALSE, TRUE, TRUE]"""
    pivots = [jnp.array([[1.0]]), jnp.array([[2.0]]), jnp.array([[3.0]])]
    predicates = [jnp.array([[False]]), jnp.array([[True]]), jnp.array([[True]])]

    new_l, new_r, any_false, any_true = find_new_bounds_unrolled(pivots, predicates)

    print("Test 2: predicates = [FALSE, TRUE, TRUE]")
    print(f"  pivots = [1.0, 2.0, 3.0]")
    print(f"  Expected: new_l=1.0 (last FALSE), new_r=2.0 (first TRUE)")
    print(f"  Got: new_l={float(new_l[0,0])}, new_r={float(new_r[0,0])}")
    print(f"  any_false={bool(any_false)}, any_true={bool(any_true)}")

    assert float(new_l[0,0]) == 1.0, f"Expected new_l=1.0, got {float(new_l[0,0])}"
    assert float(new_r[0,0]) == 2.0, f"Expected new_r=2.0, got {float(new_r[0,0])}"
    print("  ✓ PASSED\n")


def test_case_3():
    """Test case: [FALSE, FALSE, FALSE]"""
    pivots = [jnp.array([[1.0]]), jnp.array([[2.0]]), jnp.array([[3.0]])]
    predicates = [jnp.array([[False]]), jnp.array([[False]]), jnp.array([[False]])]

    new_l, new_r, any_false, any_true = find_new_bounds_unrolled(pivots, predicates)

    print("Test 3: predicates = [FALSE, FALSE, FALSE]")
    print(f"  pivots = [1.0, 2.0, 3.0]")
    print(f"  Expected: new_l=3.0 (last FALSE), any_true=False")
    print(f"  Got: new_l={float(new_l[0,0])}, new_r={float(new_r[0,0])}")
    print(f"  any_false={bool(any_false)}, any_true={bool(any_true)}")

    assert float(new_l[0,0]) == 3.0, f"Expected new_l=3.0, got {float(new_l[0,0])}"
    assert bool(any_false) == True
    assert bool(any_true) == False
    print("  ✓ PASSED\n")


def test_case_4():
    """Test case: [TRUE, TRUE, TRUE]"""
    pivots = [jnp.array([[1.0]]), jnp.array([[2.0]]), jnp.array([[3.0]])]
    predicates = [jnp.array([[True]]), jnp.array([[True]]), jnp.array([[True]])]

    new_l, new_r, any_false, any_true = find_new_bounds_unrolled(pivots, predicates)

    print("Test 4: predicates = [TRUE, TRUE, TRUE]")
    print(f"  pivots = [1.0, 2.0, 3.0]")
    print(f"  Expected: new_r=1.0 (first TRUE), any_false=False")
    print(f"  Got: new_l={float(new_l[0,0])}, new_r={float(new_r[0,0])}")
    print(f"  any_false={bool(any_false)}, any_true={bool(any_true)}")

    assert float(new_r[0,0]) == 1.0, f"Expected new_r=1.0, got {float(new_r[0,0])}"
    assert bool(any_false) == False
    assert bool(any_true) == True
    print("  ✓ PASSED\n")


if __name__ == "__main__":
    print("="*60)
    print("Testing find_new_bounds_unrolled")
    print("="*60 + "\n")

    test_case_1()
    test_case_2()
    test_case_3()
    test_case_4()

    print("="*60)
    print("✓ All tests passed!")
    print("="*60)
