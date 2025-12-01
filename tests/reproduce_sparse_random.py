
import jax
import jax.numpy as jnp
from tallax.tax.sparse_random import sparse_random_categorical

def test_repro():
    key = jax.random.PRNGKey(0)
    # shape (1, 2) to match usage key_ref[0,0]
    key_ref = jnp.array([key], dtype=jnp.uint32)

    logits = jnp.zeros((4, 10), dtype=jnp.float32)
    rows = jnp.repeat(jnp.arange(4)[:, None], 10, axis=1)
    cols = jnp.repeat(jnp.arange(10)[None, :], 4, axis=0)
    indices = (rows, cols)
    dim1_size = 10

    try:
        out = sparse_random_categorical(key_ref, logits, indices, dim1_size)
        print("Success")
    except Exception as e:
        print(f"Caught expected exception: {e}")

if __name__ == "__main__":
    test_repro()
