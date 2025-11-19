
import functools

import jax
import jax.numpy as jnp

from tallax import topk_pallas


k = 64
num_queries = 32
vocab_size = 2048

logit_key, key_act, key_weight = jax.random.split(jax.random.key(0), 3)
logits = jax.random.normal(
    key_weight, (num_queries, vocab_size), dtype=jnp.float32
).astype(jnp.bfloat16)

topk_xla = jax.jit(jax.lax.top_k, static_argnames=("k",))

def tests():
  print('topk', logits.shape, logits.dtype, k)
  print("XLA: ", topk_xla(logits, k=k))
  print("\nPallas:", topk_pallas(logits, k=k, block_size=8))
  print(
  [
  (topk_xla(logits, k=k)[i] == topk_pallas(logits, k=k, block_size=8)[i]).mean() for i in range(2)
  ]
  )

if __name__ == "__main__":
  tests()
