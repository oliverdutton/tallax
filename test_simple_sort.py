#!/usr/bin/env python3
"""Simple test to debug bitonic sort."""

import jax
import jax.numpy as jnp
from tallax._src.bitonic_sort import bitonic_sort


# Create simple test data
x = jnp.array([[3., 1., 4., 2., 5., 7., 6., 8.]], dtype=jnp.float32)
print("Input:", x)

# Test ascending sort with unroll_stages
result = bitonic_sort(x, descending=False, unroll_stages=True, interpret=True)
print("Output (unroll_stages=True):", result[0])

# Expected
expected = jnp.sort(x, axis=1)
print("Expected:", expected)

# Check
is_correct = jnp.allclose(result[0], expected)
print("Is correct:", is_correct)
