
import functools

import jax
import jax.numpy as jnp

from tallax import tax
from tallax.utils import is_cpu_platform


k = 64
num_queries = 32
vocab_size = 2048

logit_key, key_act, key_weight = jax.random.split(jax.random.key(0), 3)
logits = jax.random.normal(
    key_weight, (num_queries, vocab_size), dtype=jnp.float32
).astype(jnp.bfloat16)

topk_xla = jax.jit(jax.lax.top_k, static_argnames=("k",))

def tests():
  interpret = is_cpu_platform()
  print('topk', logits.shape, logits.dtype, k)
  print("XLA: ", topk_xla(logits, k=k))
  print("\nPallas:", tax.top_k(logits, k=k, block_size=8, interpret=interpret))
  print(
  [
  (topk_xla(logits, k=k)[i] == tax.top_k(logits, k=k, block_size=8, interpret=interpret)[i]).mean() for i in range(2)
  ]
  )

if __name__ == "__main__":
  tests()
