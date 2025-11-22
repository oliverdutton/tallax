import numpy as np
from scipy.special import factorial

def compute_exact_probs_poly(K, num_lanes):
    """
    Computes the probability distribution of the maximum depth required
    using Polynomial Generating Functions.
    Returns an array where index i corresponds to depth i+1.
    """
    # Basic constraints for float64 stability
    if K > 170:
        raise ValueError(f"K={K} is too large for standard float64 factorial. Use log-arithmetic.")

    fact_k = factorial(K)
    denom = float(num_lanes)**K
    
    probs = np.zeros(K)
    prev_cdf_val = 0.0
    
    # Optimization: Stop when CDF approaches 1.0
    # We start loop at 1 because depth 0 is impossible for K > 0
    for m in range(1, K + 1):
        # 1. Create coeffs for P_m(x) = sum(x^i / i!)
        coeffs = 1.0 / factorial(np.arange(m + 1))
        
        # 2. Binary Exponentiation for (P_m(x))^N
        current_poly = coeffs
        result_poly = np.array([1.0])
        power = num_lanes
        
        while power > 0:
            if power % 2 == 1:
                result_poly = np.convolve(result_poly, current_poly)
                if len(result_poly) > K + 1:
                    result_poly = result_poly[:K+1]
            
            current_poly = np.convolve(current_poly, current_poly)
            if len(current_poly) > K + 1:
                current_poly = current_poly[:K+1]
            
            power //= 2
            
        # 3. Extract coefficient of x^K
        if len(result_poly) <= K:
            coef_xk = 0.0
        else:
            coef_xk = result_poly[K]
            
        # 4. Calculate CDF
        current_cdf_val = (coef_xk * fact_k) / denom
        
        # 5. Calculate PDF (prob at exactly this depth)
        # Clamp to 0 to handle tiny floating point jitter
        prob_at_m = max(0.0, current_cdf_val - prev_cdf_val)
        
        probs[m-1] = prob_at_m
        prev_cdf_val = current_cdf_val
        
        # Break if we have covered the total probability mass
        if 1.0 - current_cdf_val < 1e-15:
            break
            
    return probs

def run_batch_analysis(k_values, lanes, num_blocks=8):
    """
    Runs the probability analysis for a list of K values.
    """
    for K in k_values:
        print(f"\n{'='*60}")
        print(f" SCENARIO: K={K} items -> {lanes} Lanes")
        print(f"{'='*60}")
        
        pdf = compute_exact_probs_poly(K, lanes)
        cdf = pdf.cumsum()
        # Calculate yield for 'num_blocks' independent units running simultaneously
        block_yield = cdf ** num_blocks
        
        print(f"{'Depth':<6} | {'PDF':<12} | {'CDF':<12} | {'8-Block Yield':<15}")
        print("-" * 55)
        
        # Find relevant indices (where probability is non-negligible)
        # We also ensure we print at least until yield hits ~100%
        relevant_indices = [i for i, p in enumerate(pdf) if p > 1e-9 or (i > 0 and cdf[i-1] < 0.999)]
        
        for i in relevant_indices:
            depth = i + 1
            is_safe = "<<" if block_yield[i] > 0.999 else ""
            
            print(f"{depth:<6} | {pdf[i]:.6f}     | {cdf[i]:.6f}     | {block_yield[i]:.4%} {is_safe}")
            
        avg_depth = np.sum(pdf * np.arange(1, len(pdf)+1))
        print(f"\nAverage Max Depth: {avg_depth:.4f}")

# --- Execution ---
K_VALUES = [32, 64, 128]
FIXED_LANES = 128

run_batch_analysis(K_VALUES, FIXED_LANES)
