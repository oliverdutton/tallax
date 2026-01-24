from tallax.tax.bitonic.topk import max_arrays
from tallax.vllm.topk_mask import topk_mask_ref_inputs
from tallax.vllm.topp_mask import topp_mask
from tallax.tax.sparse_random import sparse_random_categorical

@functools.partial(
  jax.jit,
)
def topk_topp_mask_and_sample(
  logits_ref,
  rng_key_ref,
  k_ref,
  p_ref,
  temperature_ref,
  dim0_offset_ref,
  sampled_tokens_ref,
  *,
  stable: bool,
  replace_val: float,
):
  token_idx = lax.broadcasted_iota(jnp.int32, logits_ref.shape, 0)
  greedy_sampled = max_arrays(
    [logits_ref[...], token_idx], num_keys=1+int(stable), axis=1
  )[1]

  logits = topk_mask_ref_inputs(logits_ref, k_ref, replace_val=replace_val, stable=stable)
  logits = topp_mask(
    logits, p_ref[...], replace_val=replace_val
  )
  logits = logits / temperature_ref[...].astype(logits.dtype)

  # random key splitting is based on idx in ravelled array
  # we pass in (batch_idx.T, token_idx.T) and sample across axis 0, taking the token_idx
  batch_idx = lax.broadcasted_iota(jnp.int32, logits.shape, 1) + dim0_offset_ref[0]
  next_tokens = sparse_random_categorical(
    rng_key_ref,
    logits,
    # these are both transposed, (token, batch) shape
    (batch_idx, token_idx),
    dim1_size=logits.shape[1],
    axis=0,
    dtype=jnp.float32,
    # take sampled_indices[1], the token idx
  )[1]

  sampled_tokens_ref[...] = jnp.where(temperature_ref[...] < _SAMPLING_EPS, greedy_sampled, next_tokens)