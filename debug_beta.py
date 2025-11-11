"""Debug script to check if beta-neutral constraint works."""
import numpy as np
from src.knockoff_neutralized.portfolio_optimizer import PortfolioOptimizer

# Simple test case: 5 assets with known betas
n_assets = 5
betas = np.array([0.8, 1.0, 1.2, 1.5, 0.9])  # Different beta values

# Strong differential alpha signal (prefer low beta assets)
alpha = np.array([5.0, 3.0, 1.0, 0.5, 4.0])  # Higher values = more attractive

# Test 1: WITHOUT beta-neutral constraint
print("=== Test 1: Standard Optimizer (no beta-neutral) ===")
opt_standard = PortfolioOptimizer(
    risk_aversion=0.1,
    beta_neutral=False,
    neutralize_factors=False,
    long_only=True,
    max_leverage=1.0,  # For long-only, leverage = 1
    max_position_size=1.0,  # Allow concentration for this test
)
result_standard = opt_standard.optimize(
    alpha_scores=alpha,
    factor_exposures=betas.reshape(-1, 1),  # Must provide even if not using
    covariance_matrix=np.eye(n_assets) * 0.01,
    current_weights=np.zeros(n_assets),
)
weights_standard = result_standard.weights
beta_exp_standard = weights_standard @ betas
print(f"Solver status: {result_standard.status}")
print(f"Weights: {weights_standard}")
print(f"Weights sum: {weights_standard.sum():.6f} (should be 1.0 for long-only)")
print(f"Beta Exposure: {beta_exp_standard:.6f}")
print(f"Expected: close to mean(betas) ≈ {betas.mean():.3f}\n")

# Test 2: WITH beta-neutral constraint (LONG-SHORT mode)
print("=== Test 2: Beta-Neutral Optimizer (Long-Short) ===")
opt_neutral = PortfolioOptimizer(
    risk_aversion=0.1,
    beta_neutral=True,
    neutralize_factors=True,
    long_only=False,  # MUST allow shorting for beta-neutrality!
    max_leverage=2.0,  # Can have long + short = 2x leverage
    max_position_size=0.5,  # Limit individual positions
)
result_neutral = opt_neutral.optimize(
    alpha_scores=alpha,
    factor_exposures=betas.reshape(-1, 1),  # MarketBeta as single factor
    covariance_matrix=np.eye(n_assets) * 0.01,
    current_weights=np.zeros(n_assets),
)
weights_neutral = result_neutral.weights
beta_exp_neutral = weights_neutral @ betas
print(f"Solver status: {result_neutral.status}")
print(f"Weights: {weights_neutral}")
print(f"Weights sum: {weights_neutral.sum():.6f} (should be 1.0)")
print(f"Beta Exposure: {beta_exp_neutral:.10f}")
print(f"Expected: ≈ 0 (within 1e-6)\n")

# Check if constraint was satisfied
if abs(beta_exp_neutral) < 1e-5:
    print("✓ Beta-neutral constraint IS working!")
else:
    print(f"✗ Beta-neutral constraint NOT working! Beta = {beta_exp_neutral:.6f} != 0")
