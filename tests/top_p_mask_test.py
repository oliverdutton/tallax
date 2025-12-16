import jax
import jax.numpy as jnp
import numpy as np
from tallax._src.top_p_and_sample import top_p_mask
from tallax._src.tpu_inference_sampling_as_standalone_file import topp_mask


def test_top_p_mask(shape, seed):
    """Test top_p_mask for exact match against topp_mask from tpu_inference_sampling.

    Strategy:
    1. Generate random p values per batch element
    2. Sort input and get argsort indices (axis=1)
    3. Apply top_p_mask to sorted input
    4. Reverse argsort to return to original order
    5. Apply topp_mask to unsorted input (per-batch element)
    6. Compare results (should match exactly in f32)
    """
    key = jax.random.key(seed)
    key, logits_key, p_key = jax.random.split(key, 3)

    # Generate random logits (f32)
    logits = jax.random.normal(logits_key, shape, dtype=jnp.float32)

    # Generate random p values from 0 to 1, different for each batch element
    p_array = jax.random.uniform(p_key, shape[:1], dtype=jnp.float32)

    replace_val = -1e12

    # Sort logits in descending order and get indices (axis=1)
    sort_indices = jnp.argsort(logits, axis=1, descending=True)
    sorted_logits = jnp.take_along_axis(logits, sort_indices, axis=1)

    # Transpose for top_p_mask (expects axis=0)
    sorted_logits_transposed = sorted_logits.T

    # Apply top_p_mask (axis=0 on transposed logits)
    result_top_p_mask_sorted = top_p_mask(
        topk_logits=sorted_logits_transposed,
        p=p_array,
        replace_val=replace_val,
        axis=0
    )

    # Transpose back to (batch, vocab)
    result_top_p_mask_sorted = result_top_p_mask_sorted.T

    # Reverse the argsort to get back to original order
    # Create inverse permutation
    inverse_sort_indices = jnp.argsort(sort_indices, axis=1)
    result_top_p_mask_original_order = jnp.take_along_axis(
        result_top_p_mask_sorted, inverse_sort_indices, axis=1
    )

    # Apply topp_mask per batch element (expects unsorted logits and scalar p)
    result_topp_mask = jnp.zeros_like(logits)
    for i in range(shape[0]):
        result_topp_mask = result_topp_mask.at[i].set(
            topp_mask(logits[i:i+1], float(p_array[i]), replace_val)[0]
        )

    # Compare results in original order (should match exactly in f32)
    np.testing.assert_array_equal(result_top_p_mask_original_order, result_topp_mask,
        err_msg=f"top_p_mask should match topp_mask for shape={shape}, seed={seed}")


if __name__ == "__main__":
    print("Running top_p_mask tests...")
    # Smaller shapes to avoid slow per-batch loops with topp_mask
    shapes = [(8, 128), (16, 256), (13, 167), (32, 128)]
    seeds = [42, 123, 456, 789, 321]

    for shape in shapes:
        for seed in seeds:
            print(f"Testing shape={shape}, seed={seed}...")
            test_top_p_mask(shape, seed)
            print(f"  ✓ Passed")

    print("\nAll top_p_mask tests passed!")
