import jax
import jax.numpy as jnp
from tallax import tax
from tallax.test_utils import verify_topk_output

def test_manual():
    print("Testing bitonic_topk on CPU...")
    k = 128
    num_tokens = 1
    vocab_size = 128

    key = jax.random.key(42)
    x = -jax.random.permutation(key, num_tokens * vocab_size).reshape(num_tokens, vocab_size).astype(jnp.int32)
    indices = jax.lax.broadcasted_iota(jnp.int32, (num_tokens, vocab_size), 1)

    result = tax.bitonic_topk((x, indices), k=k, num_keys=1, descending=True, interpret=True)

    validation = verify_topk_output(x, result)
    if validation.all():
        print("PASS")
    else:
        print("FAIL")
        print(result)

if __name__ == "__main__":
    test_manual()
