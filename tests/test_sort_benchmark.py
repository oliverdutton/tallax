
import functools
import jax
import jax.numpy as jnp
import sys
import os

# Import benchmark utils
sys.path.append(os.path.dirname(__file__))
from benchmark_utils import benchmark

from tallax import lax_sort_pallas
from tallax.utils import is_cpu_platform
from tests.test_sort_correctness import _equiv_xla_based_sort

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
                  lax_sort_pallas(operands, num_keys=num_keys, interpret=interpret, **kwargs),
                  _equiv_xla_based_sort(operands, num_keys=num_keys, **kwargs)
              )
            benchmark(_run)

if __name__ == "__main__":
  run_benchmarks()
