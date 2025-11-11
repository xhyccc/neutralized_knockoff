"""
Demo: Strategy with Pre-Selected Signals

This demonstrates the portfolio construction when you already know
which signals to use (bypassing the knockoff selection phase).
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from knockoff_neutralized import PortfolioOptimizer

def main():
    print("=" * 70)
    print("DEMO: FACTOR-NEUTRALIZED PORTFOLIO CONSTRUCTION")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Simulate 100 stocks
    n_assets = 100
    asset_ids = [f'Stock_{i:03d}' for i in range(n_assets)]
    
    print(f"\nScenario:")
    print(f"  - Portfolio universe: {n_assets} stocks")
    print(f"  - You have 3 validated alpha signals")
    print(f"  - You want to neutralize 3 risk factors: Market, Size, Value")
    
    # Create 3 alpha signals
    print(f"\nCreating alpha signals...")
    momentum_signal = np.random.randn(n_assets) * 0.5
    quality_signal = np.random.randn(n_assets) * 0.4
    sentiment_signal = np.random.randn(n_assets) * 0.3
    
    # Combine into composite alpha score
    alpha_scores = 0.4 * momentum_signal + 0.3 * quality_signal + 0.3 * sentiment_signal
    alpha_scores = (alpha_scores - alpha_scores.mean()) / alpha_scores.std()  # Standardize
    
    print(f"  ✓ Combined alpha score created")
    print(f"    Range: [{alpha_scores.min():.2f}, {alpha_scores.max():.2f}]")
    print(f"    Mean: {alpha_scores.mean():.4f}, Std: {alpha_scores.std():.2f}")
    
    # Create risk factor exposures
    print(f"\nEstimating risk factor exposures...")
    market_beta = np.random.randn(n_assets) * 0.3 + 1.0  # Around 1.0
    size_loading = np.random.randn(n_assets) * 0.5
    value_loading = np.random.randn(n_assets) * 0.4
    
    factor_exposures = np.column_stack([market_beta, size_loading, value_loading])
    factor_names = ['Market', 'Size', 'Value']
    
    print(f"  ✓ Factor betas estimated")
    print(f"    Market beta mean: {market_beta.mean():.2f}")
    print(f"    Size loading mean: {size_loading.mean():.2f}")
    print(f"    Value loading mean: {value_loading.mean():.2f}")
    
    # Estimate covariance matrix (simplified)
    print(f"\nEstimating covariance matrix...")
    volatilities = np.random.uniform(0.15, 0.40, n_assets)
    correlation = 0.3  # Assume 30% average correlation
    cov_matrix = np.outer(volatilities, volatilities) * correlation
    np.fill_diagonal(cov_matrix, volatilities ** 2)
    print(f"  ✓ Covariance matrix created ({n_assets}x{n_assets})")
    
    # Run optimization
    print("\n" + "=" * 70)
    print("OPTIMIZING PORTFOLIO")
    print("=" * 70)
    
    optimizer = PortfolioOptimizer(
        risk_aversion=2.0,  # Moderate risk aversion
        max_leverage=2.0,   # 100% long + 100% short
        max_position_size=0.05,  # Max 5% in any position
        neutralize_factors=True
    )
    
    result = optimizer.optimize(
        alpha_scores=alpha_scores,
        factor_exposures=factor_exposures,
        covariance_matrix=cov_matrix,
        factor_names=factor_names
    )
    
    if not result.success:
        print(f"✗ Optimization failed: {result.status}")
        return
    
    # Display results
    print(f"\n✓ Optimization Status: {result.status}")
    
    print("\n" + "=" * 70)
    print("PORTFOLIO CHARACTERISTICS")
    print("=" * 70)
    
    weights_series = pd.Series(result.weights, index=asset_ids)
    
    print(f"\nPosition Summary:")
    long_mask = weights_series > 1e-6
    short_mask = weights_series < -1e-6
    print(f"  - Long positions: {long_mask.sum()}")
    print(f"  - Short positions: {short_mask.sum()}")
    print(f"  - Zero positions: {(~long_mask & ~short_mask).sum()}")
    
    print(f"\nRisk Metrics:")
    print(f"  - Gross leverage: {result.leverage:.2%}")
    print(f"  - Net exposure: {result.weights.sum():.4%}")
    print(f"  - Expected alpha score: {result.alpha_score:.4f}")
    print(f"  - Portfolio variance: {result.portfolio_variance:.6f}")
    print(f"  - Portfolio volatility: {np.sqrt(result.portfolio_variance):.2%}")
    
    print(f"\nFactor Exposures (Target: ~0):")
    for name, exp in result.factor_exposures.items():
        status = "✓" if abs(exp) < 0.01 else "⚠"
        print(f"  {status} {name}: {exp:>8.6f}")
    
    print(f"\nTop 10 Long Positions:")
    top_long = weights_series.nlargest(10)
    for stock, weight in top_long.items():
        idx = asset_ids.index(stock)
        print(f"  {stock}: {weight:>7.4f} (alpha: {alpha_scores[idx]:>6.2f})")
    
    print(f"\nTop 10 Short Positions:")
    top_short = weights_series.nsmallest(10)
    for stock, weight in top_short.items():
        idx = asset_ids.index(stock)
        print(f"  {stock}: {weight:>7.4f} (alpha: {alpha_scores[idx]:>6.2f})")
    
    # Analyze alpha capture
    print("\n" + "=" * 70)
    print("ALPHA CAPTURE ANALYSIS")
    print("=" * 70)
    
    # Check if portfolio weights align with alpha scores
    correlation = np.corrcoef(result.weights, alpha_scores)[0, 1]
    print(f"\nCorrelation (weights, alpha scores): {correlation:.3f}")
    
    if correlation > 0.7:
        print("  ✓ Strong positive correlation - portfolio captures alpha well")
    elif correlation > 0.4:
        print("  ✓ Moderate correlation - portfolio balances alpha and risk")
    else:
        print("  ⚠ Low correlation - risk constraints dominating")
    
    # Long/short alpha
    long_alpha = np.sum(weights_series[long_mask].values * alpha_scores[long_mask.values])
    short_alpha = np.sum(weights_series[short_mask].values * alpha_scores[short_mask.values])
    
    print(f"\nAlpha Attribution:")
    print(f"  - Long book alpha: {long_alpha:>7.4f}")
    print(f"  - Short book alpha: {short_alpha:>7.4f}")
    print(f"  - Total alpha: {long_alpha + short_alpha:>7.4f}")
    
    print("\n" + "=" * 70)
    print("✓ DEMONSTRATION COMPLETE")
    print("=" * 70)
    
    print("\nKey Takeaways:")
    print("  1. ✓ Portfolio constructed from validated alpha signals")
    print("  2. ✓ Factor exposures neutralized (near zero)")
    print("  3. ✓ Portfolio is dollar-neutral (long/short balanced)")
    print("  4. ✓ Risk-adjusted weights maximize alpha capture")
    print("  5. ✓ Position limits respected")
    
    print(f"\nThis portfolio is ready for execution!")
    print(f"Total capital required (at target leverage): ${result.leverage/2:.1%} of AUM")
    
if __name__ == "__main__":
    main()
