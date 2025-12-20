"""
Convergence theory for divide-and-filter top-k algorithm.

This module provides two different approaches to compute recall guarantees:

1. **Probability-based approach** (compute_depth_probs, calculate_depth_thresholds):
   - Computes P(all k elements captured at depth m)
   - Used when you want probabilistic guarantees that ALL top-k elements are found
   - Example: "With 95% probability, we capture all k elements"

2. **Expected recall approach** (compute_expected_recall_at_depth, compute_required_depth_for_recall_target):
   - Computes E[number of top-k elements captured at depth m]
   - Used when you want guarantees on the average number of elements found
   - Example: "On average, we capture 0.95*k elements"

The expected recall approach is typically more favorable because it allows for
occasional misses while guaranteeing the average quality, whereas the probability
approach requires capturing ALL elements with high probability.

Example usage:
    # Approach 1: Probability-based
    >>> pdf = compute_depth_probs(k=32, num_bins=128)
    >>> cdf = pdf.cumsum()
    >>> prob_at_depth_2 = cdf[1]  # P(all k elements in first 2 bins)

    # Approach 2: Expected recall
    >>> expected_recall = compute_expected_recall_at_depth(m=2, k=32, num_bins=128)
    >>> # Returns expected number of top-k elements captured (e.g., 30.5 out of 32)

    >>> required_depth = compute_required_depth_for_recall_target(
    ...     k=32, num_bins=128, recall_target=0.95
    ... )
    >>> # Returns minimum depth m such that E[recall] >= 0.95 * k
"""

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


def compute_cdf_at_depth(m, j, num_bins):
  """
  Compute P(all j elements are in the first m bins).
  This is the CDF at depth m for j elements.

  Args:
      m: Depth (1-indexed)
      j: Number of elements to find
      num_bins: Number of bins used in partitioning

  Returns:
      Probability that all j elements are in first m bins
  """
  if j == 0:
    return 1.0
  if m == 0:
    return 0.0
  if m >= j:
    # If we have at least j bins, we might need to compute exactly
    pass

  # Precompute constants in log domain
  log_fact_j = gammaln(j + 1)
  log_denom = j * np.log(num_bins)

  # Log coeffs for P_m(x)
  terms_count = min(m + 1, j + 1)
  log_coeffs = -gammaln(np.arange(terms_count) + 1.0)

  # Binary exponentiation in log space to compute (P_m(x))^num_bins
  log_current_poly = log_coeffs
  log_result_poly = np.array([0.0])  # log(1) = 0
  power = num_bins

  while power > 0:
    if power % 2 == 1:
      log_result_poly = log_convolve_exp_shift(
        log_result_poly, log_current_poly, j + 1
      )

    if power > 1:
      log_current_poly = log_convolve_exp_shift(
        log_current_poly, log_current_poly, j + 1
      )

    power //= 2

  # Extract coefficient of x^j
  if len(log_result_poly) <= j:
    log_coef_xj = -np.inf
  else:
    log_coef_xj = log_result_poly[j]

  # Calculate CDF
  log_cdf = log_coef_xj + log_fact_j - log_denom
  cdf_val = np.exp(log_cdf) if log_cdf > -700 else 0.0
  cdf_val = min(1.0, cdf_val)

  return cdf_val


def compute_expected_recall_at_depth(m, k, num_bins):
  """
  Compute the expected number of true top-k elements captured when sampling
  to depth m (i.e., sampling the top m*num_bins elements).

  Uses the formula: E[X] = sum_{j=1}^{k} P(X >= j)
  where X is the number of true top-k elements captured.

  Args:
      m: Depth (number of bins to sample from, 1-indexed)
      k: Number of top elements we're trying to find
      num_bins: Number of bins used in partitioning

  Returns:
      Expected recall (expected number of true top-k elements captured)
  """
  expected_recall = 0.0

  # E[X] = sum_{j=1}^{k} P(X >= j)
  # where X is the number of true top-k elements captured
  for j in range(1, k + 1):
    # P(X >= j) = P(at least j elements in first m bins)
    # = P(all j elements are in first m bins when looking for j elements)
    prob_j = compute_cdf_at_depth(m, j, num_bins)
    expected_recall += prob_j

  return expected_recall


def compute_required_depth_for_recall_target(k, num_bins, recall_target):
  """
  Compute the minimum depth m such that the expected recall at depth m
  is at least recall_target * k.

  Args:
      k: Number of top elements we're trying to find
      num_bins: Number of bins used in partitioning
      recall_target: Target recall fraction (e.g., 0.95 means we want E[recall] >= 0.95 * k)

  Returns:
      Minimum depth m required (1-indexed)
  """
  target_recall = recall_target * k

  for m in range(1, k + 1):
    expected_recall = compute_expected_recall_at_depth(m, k, num_bins)
    if expected_recall >= target_recall:
      return m

  # If we get here, even depth k is not enough (shouldn't happen in practice)
  return k


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
        depths.add(i+1)  # depth is 1-indexed (threshold)
        break

  # Return sorted tuple of thresholds
  return tuple(sorted(depths))


