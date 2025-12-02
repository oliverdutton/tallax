import functools
import jax
import jax.numpy as jnp

from tallax.test_utils import benchmark
from tallax import tax
from tallax.utils import is_cpu_platform

k = 128
num_queries = 128
vocab_size = 32768

key = jax.random.key(42)
logits = jax.random.normal(
    key, (num_queries, vocab_size), dtype=jnp.float32
).astype(jnp.bfloat16)
indices = jax.lax.broadcasted_iota(jnp.int32, (num_queries, vocab_size), 1)

topk_xla = jax.jit(jax.lax.top_k, static_argnames=("k",))

@jax.jit
def add_one(x):
  return x+1

def run_benchmarks():
  interpret = is_cpu_platform()

  def _run():
    return (
      add_one(logits),
      topk_xla(logits, k=k),
      tax.bitonic_topk((logits, indices), k=k, num_keys=1, descending=True, interpret=interpret),
    )
  benchmark(_run)

if __name__ == "__main__":
  run_benchmarks()
