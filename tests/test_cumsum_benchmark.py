
import functools
import gzip
import json
import os
from glob import glob
import tempfile

import jax
import jax.numpy as jnp
import pandas as pd

from tallax.cumsum import cumsum
from tallax.utils import is_cpu_platform

def benchmark(_run):
  """Benchmark function and print timing from profiler trace."""
  def run():
    return jax.block_until_ready(_run())

  # Warmup
  run()

  with tempfile.TemporaryDirectory() as tmpdir:
    with jax.profiler.trace(tmpdir):
      run()

    # Find trace file
    files = glob(f"{tmpdir}/plugins/profile/*/**.json.gz", recursive=True)
    if not files:
        print("No trace file generated.")
        return

    path = sorted(files, key=os.path.getmtime)[-1]
    try:
        with gzip.open(path, 'rb') as f:
            trace = json.load(f)
    except Exception as e:
        print(f"Failed to load trace: {e}")
        return

    if "traceEvents" not in trace:
        print("No traceEvents in trace.")
        return

    df = pd.DataFrame(trace["traceEvents"])
    if df.empty or 'name' not in df.columns:
        print("Trace dataframe empty or no name column.")
        return

    df = df[~df.name.isna()]
    df['name'] = df.name.apply(lambda s: s.split('(')[0])

    # We look for JIT compiled functions.
    # On CPU, they might appear as PjitFunction or jit_<name>
    mask = df.name.str.contains("jit_") | (df.name == "PjitFunction")
    res = df[mask][['name', 'dur']]

    if not res.empty:
        print(res.groupby('name').sum().reset_index().to_string(index=False))
    else:
        print("No jit functions found in trace.")

def run_benchmarks():
  shape = (8, 128)
  interpret = is_cpu_platform()

  print(f"Running on {'CPU (interpret)' if interpret else 'TPU'}")

  key = jax.random.key(0)
  x = jax.random.uniform(key, shape, dtype=jnp.float32)

  # Define jitted functions with names that will show up in trace
  @functools.partial(jax.jit, static_argnames=('m',))
  def jit_cumsum_pallas(x, m):
      with jax.named_scope("jit_cumsum_pallas"):
          return cumsum(x, m=m, interpret=interpret)

  @jax.jit
  def jit_cumsum_jax(x):
      with jax.named_scope("jit_cumsum_jax"):
          return jnp.cumsum(x, axis=1)

  # Different m values
  ms = [0, 64, 128]

  for m in ms:
      print(f"\nBenchmark cumsum {shape}, m={m}")

      print("  Pallas:")
      def _run_pallas():
          out = []
          for _ in range(5000):
              out.append(jit_cumsum_pallas(x, m))
          return out
      benchmark(_run_pallas)

      print("  JAX:")
      def _run_jax():
          out = []
          for _ in range(5000):
              out.append(jit_cumsum_jax(x))
          return out
      benchmark(_run_jax)

if __name__ == "__main__":
  run_benchmarks()
