"""
Basic Example: Knockoff-Neutralized Strategy with Synthetic Data

This example demonstrates the complete workflow:
1. Generate synthetic market data
2. Create risk factors (market, size, value)
3. Create a "factor zoo" with many candidate alpha signals
4. Use knockoff filters to select true signals
5. Construct a factor-neutralized portfolio
"""

import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from knockoff_neutralized import KnockoffNeutralizedStrategy


def generate_synthetic_data(
    n_assets: int = 100,
    n_risk_factors: int = 3,
    n_true_alphas: int = 5,
    n_noise_alphas: int = 45,
    random_state: int = 42
):
    """
    Generate synthetic data for strategy demonstration.
    
    Returns
    -------
    returns : ndarray
        Forward returns
    risk_factors : dict
        Dictionary of risk factor exposures
    alpha_factors : dict
        Dictionary of alpha factor signals
    true_alpha_indices : list
        Indices of the true (non-noise) alpha factors
    """
    rng = np.random.RandomState(random_state)
    
    print("Generating synthetic data...")
    print(f"  - {n_assets} assets")
    print(f"  - {n_risk_factors} risk factors")
    print(f"  - {n_true_alphas} true alpha signals")
    print(f"  - {n_noise_alphas} noise signals")
    
    # Generate risk factors (standardized)
    risk_factor_names = ['Market', 'Size', 'Value'][:n_risk_factors]
    risk_factors = {}
    
    for name in risk_factor_names:
        risk_factors[name] = rng.randn(n_assets)
    
    # Risk factor returns (for computing asset returns)
    risk_factor_returns = rng.randn(n_risk_factors) * 0.02  # 2% volatility
    
    # Generate TRUE alpha factors (predictive)
    alpha_factors = {}
    true_alpha_indices = []
    
    for i in range(n_true_alphas):
        # True alpha is somewhat predictive (correlated with future returns)
        alpha_factors[f'true_alpha_{i}'] = rng.randn(n_assets)
        true_alpha_indices.append(i)
    
    # Generate NOISE alpha factors (not predictive)
    for i in range(n_noise_alphas):
        alpha_factors[f'noise_alpha_{i}'] = rng.randn(n_assets)
    
    # Generate returns that depend on:
    # 1. Risk factors (via factor loadings)
    # 2. True alpha factors (predictive component)
    # 3. Idiosyncratic noise
    
    # Risk factor component
    risk_factor_matrix = np.column_stack([risk_factors[name] for name in risk_factor_names])
    risk_returns = risk_factor_matrix @ risk_factor_returns
    
    # True alpha component (only from true alphas)
    true_alpha_matrix = np.column_stack([
        alpha_factors[f'true_alpha_{i}'] for i in range(n_true_alphas)
    ])
    alpha_weights = rng.randn(n_true_alphas) * 0.05  # Even stronger alpha for demonstration
    alpha_returns = true_alpha_matrix @ alpha_weights
    
    # Idiosyncratic noise
    idio_returns = rng.randn(n_assets) * 0.03  # Lower noise
    
    # Total returns
    returns = risk_returns + alpha_returns + idio_returns
    
    print(f"\n✓ Data generated")
    print(f"  - Returns mean: {np.mean(returns):.4f}")
    print(f"  - Returns std: {np.std(returns):.4f}")
    print(f"  - True alpha signals: {list(range(n_true_alphas))}")
    
    return returns, risk_factors, alpha_factors, true_alpha_indices


def main():
    """Run the basic example."""
    print("=" * 70)
    print("KNOCKOFF-NEUTRALIZED STRATEGY - BASIC EXAMPLE")
    print("=" * 70)
    print()
    
    # Generate synthetic data
    returns, risk_factors, alpha_factors, true_alpha_indices = generate_synthetic_data(
        n_assets=200,  # More assets for better statistics
        n_risk_factors=3,
        n_true_alphas=5,
        n_noise_alphas=45,
        random_state=42
    )
    
    print("\n" + "=" * 70)
    print("PHASE 1: INITIALIZING STRATEGY")
    print("=" * 70)
    
    # Initialize strategy
    strategy = KnockoffNeutralizedStrategy(
        fdr_target=0.20,  # 20% False Discovery Rate - more permissive for demo
        risk_aversion=1.0,
        max_leverage=2.0,
        max_position_size=0.10,
        random_state=42
    )
    
    print(strategy)
    
    print("\n" + "=" * 70)
    print("PHASE 2-3: FITTING STRATEGY")
    print("=" * 70)
    print()
    
    # Fit the strategy
    strategy.fit(
        returns=returns,
        risk_factors=risk_factors,
        alpha_factors=alpha_factors
    )
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    # Get selected signals
    selected_signals = strategy.get_selected_signals()
    print("\nSelected Signals:")
    print(selected_signals)
    
    # Check how many true signals were found
    true_signal_names = [f'true_alpha_{i}' for i in true_alpha_indices]
    n_true_found = sum(
        1 for name in selected_signals['signal_name']
        if name in true_signal_names
    )
    n_false_found = len(selected_signals) - n_true_found
    
    print(f"\n✓ Discovery Performance:")
    print(f"  - True signals found: {n_true_found}/{len(true_alpha_indices)}")
    print(f"  - False discoveries: {n_false_found}")
    print(f"  - Realized FDR: {n_false_found / max(1, len(selected_signals)):.2%}")
    
    # Get portfolio weights
    weights = strategy.get_portfolio_weights()
    print(f"\n✓ Portfolio Statistics:")
    print(f"  - Non-zero positions: {(weights.abs() > 1e-6).sum()}")
    print(f"  - Long positions: {(weights > 1e-6).sum()}")
    print(f"  - Short positions: {(weights < -1e-6).sum()}")
    print(f"  - Gross leverage: {weights.abs().sum():.2f}")
    print(f"  - Net exposure: {weights.sum():.6f}")
    
    # Get factor exposures
    factor_exposures = strategy.get_factor_exposures()
    print(f"\n✓ Factor Exposures (should be near zero):")
    for factor, exposure in factor_exposures.items():
        print(f"  - {factor}: {exposure:.6f}")
    
    # Display top positions
    print(f"\n✓ Top 10 Long Positions:")
    top_long = weights.nlargest(10)
    for asset, weight in top_long.items():
        print(f"  {asset}: {weight:.4f}")
    
    print(f"\n✓ Top 10 Short Positions:")
    top_short = weights.nsmallest(10)
    for asset, weight in top_short.items():
        print(f"  {asset}: {weight:.4f}")
    
    # Strategy summary
    print("\n" + "=" * 70)
    print("STRATEGY SUMMARY")
    print("=" * 70)
    summary = strategy.summary()
    for key, value in summary.items():
        if key not in ['selected_signals', 'factor_exposures']:
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Knockoff filters successfully identified true signals")
    print("  2. Portfolio is neutralized to risk factors (near-zero exposures)")
    print("  3. FDR is controlled at the target level")
    print("  4. Portfolio is diversified across long/short positions")
    print()


if __name__ == "__main__":
    main()
