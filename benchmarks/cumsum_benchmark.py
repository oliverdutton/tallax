
import functools
import jax
import jax.numpy as jnp

# Import benchmark utils
from tallax.test_utils import benchmark

from tallax import tax
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
          return tax.cumsum(x, m=m, interpret=interpret)

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
          return jit_cumsum_pallas(x, m)
      benchmark(_run_pallas)

      print("  JAX:")
      def _run_jax():
          return jit_cumsum_jax(x)
      benchmark(_run_jax)

if __name__ == "__main__":
  run_benchmarks()
