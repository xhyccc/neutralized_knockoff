"""Test if beta-neutrality is possible in long-only mode when betas have mixed signs."""
import numpy as np
from src.knockoff_neutralized.portfolio_optimizer import PortfolioOptimizer

print("="*70)
print("TEST: Beta-Neutrality with MIXED-SIGN Betas in Long-Only Mode")
print("="*70)

# Test case: Assets with BOTH positive and negative betas
n_assets = 6
betas = np.array([1.5, 1.2, 0.8, -0.3, -0.6, -0.9])  # Mixed signs!
print(f"\nAsset Betas: {betas}")
print(f"Beta range: [{betas.min():.2f}, {betas.max():.2f}]")
print(f"Mean beta: {betas.mean():.2f}")

# Differential alpha to drive optimization
alpha = np.array([3.0, 2.0, 1.5, 2.5, 3.5, 4.0])

# Test 1: Long-only WITHOUT beta-neutral
print("\n" + "="*70)
print("Test 1: Long-Only, NO beta-neutral constraint")
print("="*70)
opt_no_constraint = PortfolioOptimizer(
    risk_aversion=0.1,
    beta_neutral=False,
    neutralize_factors=False,
    long_only=True,
    max_leverage=1.0,
    max_position_size=1.0,
)

result1 = opt_no_constraint.optimize(
    alpha_scores=alpha,
    factor_exposures=betas.reshape(-1, 1),
    covariance_matrix=np.eye(n_assets) * 0.01,
    current_weights=np.zeros(n_assets),
)

weights1 = result1.weights
beta_exp1 = weights1 @ betas
print(f"Status: {result1.status}")
print(f"Weights: {weights1.round(4)}")
print(f"Weights sum: {weights1.sum():.6f}")
print(f"Beta Exposure: {beta_exp1:.6f}")

# Test 2: Long-only WITH beta-neutral (should now be feasible!)
print("\n" + "="*70)
print("Test 2: Long-Only WITH beta-neutral constraint")
print("="*70)
opt_beta_neutral = PortfolioOptimizer(
    risk_aversion=0.1,
    beta_neutral=True,
    neutralize_factors=True,
    long_only=True,
    max_leverage=1.0,
    max_position_size=1.0,
)

result2 = opt_beta_neutral.optimize(
    alpha_scores=alpha,
    factor_exposures=betas.reshape(-1, 1),
    covariance_matrix=np.eye(n_assets) * 0.01,
    current_weights=np.zeros(n_assets),
)

weights2 = result2.weights
beta_exp2 = weights2 @ betas
print(f"Status: {result2.status}")
print(f"Weights: {weights2.round(4)}")
print(f"Weights sum: {weights2.sum():.6f}")
print(f"Beta Exposure: {beta_exp2:.10f}")
print(f"Target: ≈ 0 (within 1e-6)")

if result2.status == 'optimal' and abs(beta_exp2) < 1e-5:
    print("\n✓ SUCCESS! Beta-neutrality IS achievable in long-only mode")
    print("  when assets have mixed-sign betas!")
    print("\nBreakdown:")
    for i, (w, b) in enumerate(zip(weights2, betas)):
        if w > 1e-6:
            print(f"  Asset {i}: weight={w:.4f}, beta={b:.2f}, contribution={w*b:.6f}")
else:
    print("\n✗ FAILED: Still infeasible or beta not close to zero")

# Verify: Check if we can manually construct a beta=0 portfolio
print("\n" + "="*70)
print("Mathematical Verification")
print("="*70)
print("\nCan we find weights w ≥ 0 where:")
print("  1. sum(w) = 1")
print("  2. w @ betas = 0?")

# Simple example: equal weight on positive and negative betas
pos_beta_assets = np.where(betas > 0)[0]
neg_beta_assets = np.where(betas < 0)[0]
print(f"\nPositive beta assets: {pos_beta_assets} with betas {betas[pos_beta_assets]}")
print(f"Negative beta assets: {neg_beta_assets} with betas {betas[neg_beta_assets]}")

# Try a simple manual solution: balance positive and negative
manual_weights = np.zeros(n_assets)
# Put 0.5 weight on highest negative beta (most negative = -0.9)
manual_weights[5] = 0.5  # beta = -0.9
# Put 0.5 weight on a positive beta that balances it
# Need: 0.5 * (-0.9) + 0.5 * beta_pos = 0
# => beta_pos = 0.9
# We have beta=0.8 (close), so use a mix
manual_weights[2] = 0.5  # beta = 0.8
manual_beta = manual_weights @ betas
print(f"\nManual attempt: weights={manual_weights}")
print(f"  Sum: {manual_weights.sum()}")
print(f"  Beta: {manual_beta:.4f}")

# Better manual solution using linear algebra
# We want to allocate between one positive and one negative beta asset
# For betas [0.8, -0.9], to get beta=0: w1*0.8 + w2*(-0.9) = 0, w1+w2=1
# => w1 = 0.9/(0.8+0.9) = 0.9/1.7 ≈ 0.529
w_pos = 0.9 / (0.8 + 0.9)
w_neg = 0.8 / (0.8 + 0.9)
manual_weights2 = np.zeros(n_assets)
manual_weights2[2] = w_pos  # beta = 0.8
manual_weights2[5] = w_neg  # beta = -0.9
manual_beta2 = manual_weights2 @ betas
print(f"\nBetter manual solution: weights={manual_weights2.round(4)}")
print(f"  Sum: {manual_weights2.sum():.6f}")
print(f"  Beta: {manual_beta2:.10f}")
print("\n✓ Yes! We can manually construct beta=0 with long-only!")
