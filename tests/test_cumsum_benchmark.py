
import functools
import jax
import jax.numpy as jnp
import sys
import os

# Import benchmark utils
sys.path.append(os.path.dirname(__file__))
from benchmark_utils import benchmark

from tallax.cumsum import lax_cumsum_pallas
from tallax.utils import is_cpu_platform

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
          return lax_cumsum_pallas(x, m=m, interpret=interpret)

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
