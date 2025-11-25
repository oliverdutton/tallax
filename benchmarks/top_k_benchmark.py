
import functools
import jax
import jax.numpy as jnp

# Import benchmark utils
from tallax.test_utils import benchmark

from tallax import tax
from tallax.utils import is_cpu_platform

k = 64
num_queries = 32
vocab_size = 201088
hidden_dim = 2880

logit_key, key_act, key_weight = jax.random.split(jax.random.key(0), 3)
x = jax.random.normal(key_act, (num_queries, hidden_dim), dtype=jnp.bfloat16)
w = jax.random.normal(key_weight, (hidden_dim, vocab_size), dtype=jnp.bfloat16)
logits = jax.random.normal(
    key_weight, (num_queries, vocab_size), dtype=jnp.float32
).astype(jnp.bfloat16)

topk_xla = jax.jit(jax.lax.top_k, static_argnames=("k",))
approx_topk_xla = jax.jit(jax.lax.approx_max_k, static_argnames=("k",))

@jax.jit
def add_one(x):
  return x+1


@jax.jit
@functools.partial(jax.vmap, in_axes=(0, None))
def matmul_and_topk_xla(x, w, k=k):
  logits = x @ w
  return jax.lax.top_k(logits, k)

def run_benchmarks():
  interpret = is_cpu_platform()
  def _run():
    return (
      add_one(logits),
      topk_xla(logits, k=k),
      tax.top_k(logits, k=k, block_size=8, interpret=interpret),
      # Not exact. Runtime varies with recall, here run with default 0.95
      approx_topk_xla(logits, k=k),
    )
  benchmark(_run)

if __name__ == "__main__":
  run_benchmarks()
