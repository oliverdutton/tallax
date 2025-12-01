import jax
import jax.numpy as jnp
from tallax.tax.bitonic_topk import bitonic_topk
from tallax.utils import NUM_LANES
import traceback

def test_repro():
    # Test with 1 operand
    num_tokens = 8
    vocab = 256 # 2 * 128

    vals = jax.random.uniform(jax.random.PRNGKey(0), (num_tokens, vocab))

    print("Testing single operand...")
    try:
        out = bitonic_topk(vals, k=NUM_LANES, interpret=True)
        print("Single operand success")
    except Exception:
        traceback.print_exc()

    # Test with 3 operands
    print("Testing triple operand...")
    vals = jax.random.uniform(jax.random.PRNGKey(0), (num_tokens, vocab))
    idxs = jnp.broadcast_to(jnp.arange(vocab), (num_tokens, vocab))
    payload = jnp.ones_like(vals)

    try:
        out = bitonic_topk((vals, idxs, payload), k=NUM_LANES, num_keys=1, interpret=True)
        print("Triple operand success")
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    test_repro()
