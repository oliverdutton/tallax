"""Debug parallel binary search to understand the issue."""

import jax
import jax.numpy as jnp
from tallax.vllm.binary_search import binary_search, generate_pivots, find_new_bounds_unrolled


def simple_test():
    """Simple test case with known values."""

    # Test data: find threshold where exactly 3 values are greater than threshold
    # Values: [10, 8, 6, 4, 2]
    # If threshold = 5, then values > 5 are [10, 8, 6] = 3 values

    values = jnp.array([[10.0, 8.0, 6.0, 4.0, 2.0]])  # Shape (1, 5)
    k = 3

    # Predicate: count of values > threshold is < k
    # We want LARGEST threshold where this is FALSE (i.e., where count >= k)
    predicate_fn = lambda pivot: (values > pivot).sum(-1, keepdims=True) < k

    # Expected result: threshold should be around 6 (exclusive)
    # At threshold = 5.99...: count > threshold = 3 (FALSE - we want this)
    # At threshold = 6.00...: count > threshold = 2 (TRUE)

    print("Simple test case:")
    print(f"Values: {values}")
    print(f"Target k: {k}")
    print()

    # Test with num_pivots=1
    l1, r1 = binary_search(
        predicate_fn,
        jnp.array([[-jnp.inf]]),
        jnp.array([[jnp.inf]]),
        num_pivots=1,
    )

    print(f"num_pivots=1:")
    print(f"  l={float(l1[0,0]):.6f}, r={float(r1[0,0]):.6f}")
    print(f"  count(values > l) = {int((values > l1).sum())}")
    print(f"  count(values > r) = {int((values > r1).sum())}")
    print(f"  Predicate at l: {bool(predicate_fn(l1)[0,0])} (should be FALSE)")
    print(f"  Predicate at r: {bool(predicate_fn(r1)[0,0])} (should be TRUE)")
    print()

    # Test with num_pivots=3
    l3, r3 = binary_search(
        predicate_fn,
        jnp.array([[-jnp.inf]]),
        jnp.array([[jnp.inf]]),
        num_pivots=3,
    )

    print(f"num_pivots=3:")
    print(f"  l={float(l3[0,0]):.6f}, r={float(r3[0,0]):.6f}")
    print(f"  count(values > l) = {int((values > l3).sum())}")
    print(f"  count(values > r) = {int((values > r3).sum())}")
    print(f"  Predicate at l: {bool(predicate_fn(l3)[0,0])} (should be FALSE)")
    print(f"  Predicate at r: {bool(predicate_fn(r3)[0,0])} (should be TRUE)")
    print()

    # Check if they match
    print(f"Results match: l={jnp.allclose(l1, l3)}, r={jnp.allclose(r1, r3)}")


def test_find_new_bounds():
    """Test find_new_bounds_unrolled directly."""

    print("\n" + "="*60)
    print("Testing find_new_bounds_unrolled directly")
    print("="*60 + "\n")

    # Test case 1: [FALSE, FALSE, TRUE]
    pivots = [jnp.array([[1.0]]), jnp.array([[2.0]]), jnp.array([[3.0]])]
    predicates = [jnp.array([[False]]), jnp.array([[False]]), jnp.array([[True]])]

    new_l, new_r, any_false, any_true = find_new_bounds_unrolled(pivots, predicates)

    print("Test 1: predicates = [FALSE, FALSE, TRUE]")
    print(f"  pivots = [1.0, 2.0, 3.0]")
    print(f"  Expected: new_l=2.0 (last FALSE), new_r=3.0 (first TRUE)")
    print(f"  Got: new_l={float(new_l[0,0])}, new_r={float(new_r[0,0])}")
    print(f"  any_false={bool(any_false[0,0])}, any_true={bool(any_true[0,0])}")
    print()

    # Test case 2: [FALSE, TRUE, TRUE]
    predicates2 = [jnp.array([[False]]), jnp.array([[True]]), jnp.array([[True]])]
    new_l2, new_r2, any_false2, any_true2 = find_new_bounds_unrolled(pivots, predicates2)

    print("Test 2: predicates = [FALSE, TRUE, TRUE]")
    print(f"  pivots = [1.0, 2.0, 3.0]")
    print(f"  Expected: new_l=1.0 (last FALSE), new_r=2.0 (first TRUE)")
    print(f"  Got: new_l={float(new_l2[0,0])}, new_r={float(new_r2[0,0])}")
    print(f"  any_false={bool(any_false2[0,0])}, any_true={bool(any_true2[0,0])}")
    print()

    # Test case 3: [FALSE, FALSE, FALSE]
    predicates3 = [jnp.array([[False]]), jnp.array([[False]]), jnp.array([[False]])]
    new_l3, new_r3, any_false3, any_true3 = find_new_bounds_unrolled(pivots, predicates3)

    print("Test 3: predicates = [FALSE, FALSE, FALSE]")
    print(f"  pivots = [1.0, 2.0, 3.0]")
    print(f"  Expected: new_l=3.0 (last FALSE), any_true=False")
    print(f"  Got: new_l={float(new_l3[0,0])}, new_r={float(new_r3[0,0])}")
    print(f"  any_false={bool(any_false3[0,0])}, any_true={bool(any_true3[0,0])}")
    print()

    # Test case 4: [TRUE, TRUE, TRUE]
    predicates4 = [jnp.array([[True]]), jnp.array([[True]]), jnp.array([[True]])]
    new_l4, new_r4, any_false4, any_true4 = find_new_bounds_unrolled(pivots, predicates4)

    print("Test 4: predicates = [TRUE, TRUE, TRUE]")
    print(f"  pivots = [1.0, 2.0, 3.0]")
    print(f"  Expected: new_r=1.0 (first TRUE), any_false=False")
    print(f"  Got: new_l={float(new_l4[0,0])}, new_r={float(new_r4[0,0])}")
    print(f"  any_false={bool(any_false4[0,0])}, any_true={bool(any_true4[0,0])}")


if __name__ == "__main__":
    simple_test()
    test_find_new_bounds()
