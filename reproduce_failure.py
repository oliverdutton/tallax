
import jax
import jax.numpy as jnp
from jax import lax
import math
from tallax.tax.pallas_topk_mask import topk_mask_pallas
from tallax.tax.utils import NUM_LANES

def reproduce_failure():
    print("Reproducing Failure...")
    # Config from logs: batch=16, vocab=1024, k=18
    batch_size = 16
    vocab_size = 1024
    k = 18
    
    # We want to reproduce the state at Batch 11
    # The logs showed:
    # Selected Start: 724
    # Chunk Size (in kernel): ~362? (Wait, let's calculate it exactly)
    
    chunk_size_1 = int(math.sqrt(vocab_size // NUM_LANES) * NUM_LANES)
    print(f"Calculated Stage 1 Chunk Size: {chunk_size_1}")
    
    # If chunk_size_1 is 362:
    # 724 / 362 = 2. So it's the 3rd chunk (idx 2).
    
    # Let's create an input where the 18th element is in the 3rd chunk of the first stage.
    # Stage 1 Chunks: [0:362], [362:724], [724:1024]
    # We want matches in the first two chunks to be < 18.
    # Say 10 matches in [0:724].
    # Then we need 8 more matches in [724:1024].
    
    rng = jax.random.PRNGKey(999)
    # To make it easier, let's just use the same PRNG logic as debug_config_13.py
    # but only for the failing iteration.
    
    # Actually, let's just use the provided logs' values if possible.
    # The logs say Batch 11 failed.
    
    # Let's just run the original script's logic for Batch 11 specifically.
    
    # def get_config_13_data():
    rng = jax.random.PRNGKey(999)
    for i in range(13):
        rng, subkey = jax.random.split(rng)
        bs = jax.random.choice(subkey, jnp.array([4, 8, 16]), shape=()).item()
        rng, subkey = jax.random.split(rng)
        vs = jax.random.choice(subkey, jnp.array([128, 256, 512, 1024, 2048]), shape=()).item()
        rng, subkey = jax.random.split(rng)
        cur_k = jax.random.randint(subkey, shape=(), minval=1, maxval=min(100, vs)).item()
        rng, subkey = jax.random.split(rng)
        x = jax.random.uniform(subkey, (bs, vs))
        if i % 3 == 0:
            rng, subkey = jax.random.split(rng)
            granularity = jax.random.choice(subkey, jnp.array([5, 10, 20])).item()
            x = jnp.round(x * granularity) / granularity
        # return x, cur_k
        k_val = cur_k
        # for k in range(x.shape[1])
        # Focus only on Batch 14
        # for k_val in [5]: #range(1, x.shape[-1]):
          # x = x[15:16]
        print(f"Reproduction Config: x.shape={x.shape}, k={k_val}")

        topk_vals, topk_idxs = jax.lax.top_k(x, k_val)
        threshold = topk_vals[0, -1]
        all_match_indices = jnp.where(x[0] == threshold)[0]
        # print(f"Threshold: {threshold}")
        # print(f"All indices matching threshold: {all_match_indices}")
        # print(f"Total matches: {len(all_match_indices)}")
        
        # Run our Pallas implementation
        our_result = topk_mask_pallas(x, k_val, replace_val=-jnp.inf, stable=True, interpret=True)
        
        # Reference
        topk_vals, topk_idxs = jax.lax.top_k(x, k_val)
        base = jnp.full_like(our_result, -jnp.inf)
        ref_result = jax.vmap(lambda x, idx, val: x.at[idx].set(val))(base, topk_idxs, topk_vals)
        
        match = jnp.all(our_result == ref_result)
        print(f"Match: {match}")
        
        if not match:
            import sys; sys.exit(1)
            # Find where it differs
            diff_batch = jnp.where((our_result != ref_result).any(axis=1))[0]
            print(f"Mismatched batches: {diff_batch}")
            
            for b in diff_batch:
                print(f"\nBatch {b} analysis:")
                # Find the threshold
                t = topk_vals[b, -1]
                print(f"Threshold: {t}")
                
                our_matches = (our_result[b] == t).sum()
                ref_matches = (ref_result[b] == t).sum()
                print(f"Our matches for threshold: {our_matches}")
                print(f"Ref matches for threshold: {ref_matches}")
                
                # Find boundary index
                ref_indices = jnp.where(ref_result[b] == t)[0]
                our_indices = jnp.where(our_result[b] == t)[0]
                print(f"Ref indices for threshold: {ref_indices}")
                print(f"Our indices for threshold: {our_indices}")
                
                # Total elements kept
                our_kept = (our_result[b] > -jnp.inf).sum()
                print(f"Total elements kept by us: {our_kept} (expected {k_val})")

if __name__ == "__main__":
    reproduce_failure()
