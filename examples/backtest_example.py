"""
Advanced Example: Time Series Backtesting

Demonstrates how to use the strategy with time series data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from knockoff_neutralized import KnockoffNeutralizedStrategy


def generate_panel_data(
    n_periods: int = 252,
    n_assets: int = 100,
    n_risk_factors: int = 3,
    n_true_alphas: int = 5,
    n_noise_alphas: int = 20,
    random_state: int = 42
):
    """Generate panel data for backtesting."""
    rng = np.random.RandomState(random_state)
    
    print(f"Generating panel data...")
    print(f"  - {n_periods} periods")
    print(f"  - {n_assets} assets")
    print(f"  - {n_risk_factors} risk factors")
    print(f"  - {n_true_alphas} true signals, {n_noise_alphas} noise signals")
    
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
    asset_ids = [f'Asset_{i:03d}' for i in range(n_assets)]
    
    # Risk factors panel
    risk_factor_names = ['Market', 'Size', 'Value'][:n_risk_factors]
    risk_factors_panel = {}
    
    for name in risk_factor_names:
        # Generate time series with some autocorrelation
        factor_series = np.cumsum(rng.randn(n_periods, n_assets) * 0.1, axis=0)
        risk_factors_panel[name] = pd.DataFrame(
            factor_series,
            index=dates,
            columns=asset_ids
        )
    
    # Alpha factors panel
    alpha_factors_panel = {}
    true_alpha_names = []
    
    for i in range(n_true_alphas):
        name = f'true_alpha_{i}'
        true_alpha_names.append(name)
        # True alphas have some persistence
        alpha_series = np.cumsum(rng.randn(n_periods, n_assets) * 0.05, axis=0)
        alpha_factors_panel[name] = pd.DataFrame(
            alpha_series,
            index=dates,
            columns=asset_ids
        )
    
    for i in range(n_noise_alphas):
        name = f'noise_alpha_{i}'
        # Noise is just random
        alpha_series = rng.randn(n_periods, n_assets)
        alpha_factors_panel[name] = pd.DataFrame(
            alpha_series,
            index=dates,
            columns=asset_ids
        )
    
    # Generate returns
    returns_panel = pd.DataFrame(
        np.zeros((n_periods, n_assets)),
        index=dates,
        columns=asset_ids
    )
    
    for t in range(1, n_periods):
        # Risk component
        risk_component = sum(
            risk_factors_panel[name].iloc[t-1].values * rng.randn()
            for name in risk_factor_names
        ) * 0.01
        
        # True alpha component
        alpha_component = sum(
            alpha_factors_panel[name].iloc[t-1].values
            for name in true_alpha_names
        ) * 0.005
        
        # Noise
        noise = rng.randn(n_assets) * 0.02
        
        returns_panel.iloc[t] = risk_component + alpha_component + noise
    
    print("✓ Panel data generated")
    
    return returns_panel, risk_factors_panel, alpha_factors_panel, true_alpha_names


def backtest_strategy(
    returns_panel,
    risk_factors_panel,
    alpha_factors_panel,
    refit_frequency: int = 60,
    rebalance_frequency: int = 5
):
    """Run a simple backtest."""
    
    print("\nRunning backtest...")
    print(f"  - Refit signals every {refit_frequency} days")
    print(f"  - Rebalance portfolio every {rebalance_frequency} days")
    
    n_periods = len(returns_panel)
    n_assets = len(returns_panel.columns)
    
    # Storage
    portfolio_weights = pd.DataFrame(
        np.zeros((n_periods, n_assets)),
        index=returns_panel.index,
        columns=returns_panel.columns
    )
    portfolio_returns = pd.Series(0.0, index=returns_panel.index)
    
    strategy = KnockoffNeutralizedStrategy(
        fdr_target=0.15,
        risk_aversion=1.0,
        max_leverage=2.0,
        random_state=42
    )
    
    # Training period
    train_period = 60
    
    for t in range(train_period, n_periods):
        # Refit signals periodically
        if t == train_period or (t - train_period) % refit_frequency == 0:
            print(f"\n  Refitting at t={t} ({returns_panel.index[t].date()})...")
            
            # Use lookback window
            lookback = 60
            train_returns = returns_panel.iloc[t-lookback:t].mean().values
            
            train_risk = {
                name: df.iloc[t].values
                for name, df in risk_factors_panel.items()
            }
            
            train_alpha = {
                name: df.iloc[t].values
                for name, df in alpha_factors_panel.items()
            }
            
            try:
                strategy.fit(
                    returns=train_returns,
                    risk_factors=train_risk,
                    alpha_factors=train_alpha
                )
                
                n_selected = len(strategy.selected_alpha_indices_)
                print(f"    Selected {n_selected} signals")
                
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        # Rebalance periodically
        elif (t - train_period) % rebalance_frequency == 0 and strategy.is_fitted_:
            current_risk = {
                name: df.iloc[t].values
                for name, df in risk_factors_panel.items()
            }
            
            current_alpha = {
                name: df.iloc[t].values
                for name, df in alpha_factors_panel.items()
            }
            
            # Use recent returns for rebalance (small non-zero variance)
            recent_returns = returns_panel.iloc[t-5:t].mean().values + np.random.randn(n_assets) * 0.001
            
            try:
                strategy.rebalance(
                    returns=recent_returns,
                    risk_factors=current_risk,
                    alpha_factors=current_alpha,
                    refit_signals=False
                )
            except Exception as e:
                # Silently continue on rebalance errors
                pass
        
        # Store current weights
        if strategy.is_fitted_ and strategy.current_weights_ is not None:
            portfolio_weights.iloc[t] = strategy.current_weights_
        
        # Compute return
        portfolio_returns.iloc[t] = np.dot(
            portfolio_weights.iloc[t].values,
            returns_panel.iloc[t].values
        )
    
    print("\n✓ Backtest complete")
    
    return portfolio_returns, portfolio_weights


def plot_results(portfolio_returns):
    """Plot backtest results."""
    
    print("\nComputing performance metrics...")
    
    # Cumulative returns
    cum_returns = (1 + portfolio_returns).cumprod()
    
    # Metrics
    total_return = cum_returns.iloc[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
    annualized_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0
    
    max_dd = (cum_returns / cum_returns.cummax() - 1).min()
    
    print(f"\nPerformance Metrics:")
    print(f"  - Total Return: {total_return:.2%}")
    print(f"  - Annualized Return: {annualized_return:.2%}")
    print(f"  - Annualized Volatility: {annualized_vol:.2%}")
    print(f"  - Sharpe Ratio: {sharpe:.2f}")
    print(f"  - Max Drawdown: {max_dd:.2%}")
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Cumulative returns
    axes[0].plot(cum_returns.index, cum_returns.values, label='Strategy')
    axes[0].set_title('Cumulative Returns')
    axes[0].set_ylabel('Cumulative Return')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Rolling Sharpe (60-day)
    rolling_sharpe = (
        portfolio_returns.rolling(60).mean() /
        portfolio_returns.rolling(60).std()
    ) * np.sqrt(252)
    
    axes[1].plot(rolling_sharpe.index, rolling_sharpe.values, label='60-day Rolling Sharpe')
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1].set_title('Rolling Sharpe Ratio')
    axes[1].set_ylabel('Sharpe Ratio')
    axes[1].set_xlabel('Date')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=150)
    print("\n✓ Plot saved to 'backtest_results.png'")
    
    plt.show()


def main():
    """Run the advanced example."""
    print("=" * 70)
    print("KNOCKOFF-NEUTRALIZED STRATEGY - TIME SERIES BACKTEST")
    print("=" * 70)
    print()
    
    # Generate data
    returns_panel, risk_factors_panel, alpha_factors_panel, true_alpha_names = \
        generate_panel_data(
            n_periods=252,
            n_assets=100,
            n_risk_factors=3,
            n_true_alphas=5,
            n_noise_alphas=20,
            random_state=42
        )
    
    # Run backtest
    portfolio_returns, portfolio_weights = backtest_strategy(
        returns_panel,
        risk_factors_panel,
        alpha_factors_panel,
        refit_frequency=60,
        rebalance_frequency=5
    )
    
    # Plot results
    plot_results(portfolio_returns)
    
    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
