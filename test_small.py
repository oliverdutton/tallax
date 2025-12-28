#!/usr/bin/env python3
"""Simple test for small shape (3, 17)."""
import jax
import jax.numpy as jnp
from tallax import tax
from tallax._src.test_utils import verify_sort_output
from tallax._src.utils import is_cpu_platform

shape = (3, 17)
dtype = jnp.float32
num_arrays = 1
num_keys = 1

print(f"Testing shape {shape}...")

# Generate test data
key = jax.random.key(42)
operands = [jax.random.normal(key, shape, dtype=dtype)]

interpret = is_cpu_platform()

# Run sort
print("Running sort...")
outputs = tax.sort(
    operands,
    num_keys=num_keys,
    interpret=interpret
)

print("Verifying output...")
# Verify outputs
valid = verify_sort_output(
    operands,
    outputs,
    num_keys=num_keys,
    interpret=interpret
)

if valid:
    print("✓ Test PASSED")
else:
    print("✗ Test FAILED")
