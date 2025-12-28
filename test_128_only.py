#!/usr/bin/env python3
import jax
import jax.numpy as jnp
from tallax import tax
from tallax._src.test_utils import verify_sort_output
from tallax._src.utils import is_cpu_platform

shape = (8, 128)
dtype = jnp.float32
key = jax.random.key(42)
operands = [jax.random.normal(key, shape, dtype=dtype)]
interpret = is_cpu_platform()

print(f'Testing {shape}...')
outputs = tax.sort(operands, num_keys=1, interpret=interpret)
valid = verify_sort_output(operands, outputs, num_keys=1, interpret=interpret)

if valid:
    print(f'✓ Test PASSED for {shape}')
else:
    print(f'✗ Test FAILED for {shape}')
