
import functools
import jax
import jax.numpy as jnp
import sys
import os

# Add parent directory to path to import tests
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import benchmark utils (assuming running from benchmarks dir or added to path)
import benchmark_utils
from benchmark_utils import benchmark

from tallax import tax
from tallax.utils import is_cpu_platform
from tests.sort_test import _equiv_xla_based_sort

def run_benchmarks():
  ntoken = 8
  interpret = is_cpu_platform()
  for num_operands in range(1,2):
    for num_keys in range(1, num_operands+1):
      for n in (
          2**13,
          2**12+1,
      ):
        for dtype in (
            jnp.float32,
            jnp.bfloat16,
            jnp.int32,
        ):
          operands = list(jax.random.randint(jax.random.key(0), (num_operands, ntoken,n), jnp.iinfo(jnp.int32).min, jnp.iinfo(jnp.int32).max, jnp.int32).view(dtype)[...,:n])
          for kwargs in (
              dict(),
              dict(descending=True),
              dict(return_argsort=True),
              dict(is_stable=True),
          ):
            x = operands[0]
            print(f'\n{(x.shape, x.dtype)}\n{num_operands=} {num_keys=} {kwargs=}')
            def _run():
              return (
                  tax.sort(operands, num_keys=num_keys, interpret=interpret, **kwargs),
                  _equiv_xla_based_sort(operands, num_keys=num_keys, **kwargs)
              )
            benchmark(_run)

if __name__ == "__main__":
  run_benchmarks()
