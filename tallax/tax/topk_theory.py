import math
import numpy as np
from scipy.special import gammaln

def log_convolve_exp_shift(log_a, log_b, trunc_len):
  """
  Performs convolution in log-space using the Exp-Shift trick to maintain stability.
  Mathematically equivalent to: log(convolve(exp(log_a), exp(log_b)))
  """
  # 1. Shift logs to avoid overflow/underflow
  # The maximum log value becomes 0 (linear value 1.0)
  max_a = np.max(log_a)
  max_b = np.max(log_b)

  # Handle cases where arrays represent 0 probability (-inf)
  if max_a == -np.inf or max_b == -np.inf:
    return np.full(min(len(log_a) + len(log_b) - 1, trunc_len), -np.inf)

  # 2. Move to linear space safely
  lin_a = np.exp(log_a - max_a)
  lin_b = np.exp(log_b - max_b)

  # 3. Standard Convolution
  lin_conv = np.convolve(lin_a, lin_b)

  # 4. Truncate to k+1 to prevent array explosion
  if len(lin_conv) > trunc_len:
    lin_conv = lin_conv[:trunc_len]

  # 5. Move back to Log space and undo the shift
  # Use np.errstate to suppress warnings for log(0) which correctly results in -inf
  with np.errstate(divide='ignore'):
    log_conv = np.log(lin_conv)

  return log_conv + max_a + max_b

def compute_depth_probs(k, num_bins):
  """
  Computes the probability distribution of max depth.
  """
  # Precompute constants in log domain
  # log(k!) = gammaln(k + 1)
  log_fact_k = gammaln(k + 1)
  # log(num_bins^k) = k * log(num_bins)
  log_denom = k * np.log(num_bins)

  probs = np.zeros(k)
  prev_cdf_val = 0.0

  for m in range(1, k + 1):
    # 1. Log Coeffs for P_m(x): log(1/i!) = -log(i!) = -gammaln(i+1)
    # We only need terms up to min(m, k)
    terms_count = min(m + 1, k + 1)
    log_coeffs = -gammaln(np.arange(terms_count) + 1.0)

    # 2. Binary Exponentiation in Log Space
    # We calculate (P_m(x))^num_bins
    log_current_poly = log_coeffs
    log_result_poly = np.array([0.0]) # log(1) = 0
    power = num_bins

    while power > 0:
      if power % 2 == 1:
        log_result_poly = log_convolve_exp_shift(
          log_result_poly, log_current_poly, k + 1
        )

      if power > 1: # Optimization: skip last square if not needed
        log_current_poly = log_convolve_exp_shift(
          log_current_poly, log_current_poly, k + 1
        )

      power //= 2

    # 3. Extract coefficient of x^k (index k)
    if len(log_result_poly) <= k:
      log_coef_xk = -np.inf
    else:
      log_coef_xk = log_result_poly[k]

    # 4. Calculate Log CDF -> Linear CDF
    # log_prob = log_coef + log_fact - log_denom
    log_cdf = log_coef_xk + log_fact_k - log_denom

    # Clamp exp to 1.0 to handle floating point noise > 0
    current_cdf_val = np.exp(log_cdf) if log_cdf > -700 else 0.0
    current_cdf_val = min(1.0, current_cdf_val)

    # 5. Calculate PDF
    prob_at_m = max(0.0, current_cdf_val - prev_cdf_val)

    # Store in array (index m-1 corresponds to depth m)
    probs[m-1] = prob_at_m
    prev_cdf_val = current_cdf_val

    # Optimization: Break if we reached 100% probability mass
    if current_cdf_val >= 1.0 - 1e-14:
      break

  return probs


def compute_power_of_2_schedule(schedule):
  """
  Convert a schedule to power-of-2 values.

  Args:
      schedule: Tuple of depth values

  Returns:
      Tuple of power-of-2 depth values, sorted and deduplicated
  """
  if not schedule:
    return ()

  power_of_2_depths = []
  for depth in schedule:
    if depth <= 0:
      continue
    # Round up to nearest power of 2
    power_of_2_depth = 2 ** math.ceil(math.log2(depth))
    power_of_2_depths.append(power_of_2_depth)

  # Remove duplicates and sort
  return tuple(sorted(set(power_of_2_depths)))


def calculate_depth_thresholds(k, num_bins, block_size=8, target_yields=(0.66, 0.95, 0.9999)):
  """
  Calculate minimum depths needed to reach probability thresholds.
  Checks if (cdf ^ block_size) >= target.

  Returns:
      Tuple of threshold depths (0-indexed)
  """
  # Compute the probability distribution
  pdf = compute_depth_probs(k, num_bins)
  cdf = pdf.cumsum()

  # Calculate yield for the full block size
  block_yields = cdf ** block_size

  # Find minimum depth for each threshold
  depths = set()
  for threshold in target_yields:
    # Find first depth where block_yield >= threshold
    for i, val in enumerate(block_yields):
      if val >= threshold:
        depths.add(i)  # depth is 0-indexed (threshold)
        break

  # Return sorted tuple of thresholds
  return tuple(sorted(depths))


def compute_schedules(max_k: int, num_blocks: int = 128, block_size: int = 8):
  """
  Auto-compute block_topk_schedule and topk_schedule based on depth probabilities.

  Args:
      max_k: Maximum k value for top-k computation
      num_blocks: Number of blocks (lanes) for computation
      block_size: Token blocking size

  Returns:
      tuple: (block_topk_schedule, topk_schedule)
  """
  # Get optimal depth thresholds from probability analysis (0-indexed)
  thresholds = calculate_depth_thresholds(
      max_k, num_blocks, block_size=block_size
  )

  # Convert thresholds to schedules by adding 1
  block_topk_schedule = tuple(t + 1 for t in thresholds)

  # Always ensure max_k is included in block_topk_schedule
  if not block_topk_schedule or block_topk_schedule[-1] != max_k:
    block_topk_schedule = block_topk_schedule + (max_k,)

  # Compute power-of-2 schedule from block_topk_schedule for topk_schedule
  topk_schedule = compute_power_of_2_schedule(block_topk_schedule)

  # Ensure max_k is included in topk_schedule
  if not topk_schedule or topk_schedule[-1] != max_k:
    topk_schedule = topk_schedule + (max_k,)

  return block_topk_schedule, topk_schedule
