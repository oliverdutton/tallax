
import functools
import gzip
import json
import os
from glob import glob
import tempfile

import jax
import jax.numpy as jnp
import pandas as pd

from tallax import lax_sort_pallas
from tests.test_sort_correctness import _equiv_xla_based_sort

def benchmark(_run):
  """Benchmark function and print timing from profiler trace."""
  def run():
    return jax.block_until_ready(_run())

  run()
  with tempfile.TemporaryDirectory() as tmpdir:
    with jax.profiler.trace(tmpdir):
      run()

    path = sorted(
        glob(f"{tmpdir}/plugins/profile/*/**.json.gz", recursive=True),
        key=os.path.getmtime
    )[-1]
    trace = json.load(gzip.open(path))
    df = pd.DataFrame(trace["traceEvents"])
    df = df[~df.name.isna()]
    df['name'] = df.name.apply(lambda s: s.split('(')[0])
    print(df[df.name.str.contains("jit_")][['name', 'dur']].to_string(index=False))

def run_benchmarks():
  ntoken = 8
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
                  lax_sort_pallas(operands, num_keys=num_keys, **kwargs),
                  _equiv_xla_based_sort(operands, num_keys=num_keys, **kwargs)
              )
            benchmark(_run)

if __name__ == "__main__":
  run_benchmarks()
