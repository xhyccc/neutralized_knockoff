# Usage Guide

## Installation

```bash
# Clone or navigate to the project directory
cd knockoff_neutralized

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Quick Start

```python
from knockoff_neutralized import KnockoffNeutralizedStrategy
import numpy as np

# Prepare your data
n_assets = 100
returns = np.random.randn(n_assets)  # Forward returns

risk_factors = {
    'Market': np.random.randn(n_assets),
    'Size': np.random.randn(n_assets),
    'Value': np.random.randn(n_assets)
}

alpha_factors = {
    f'Signal_{i}': np.random.randn(n_assets) 
    for i in range(50)  # Your factor zoo
}

# Initialize and fit strategy
strategy = KnockoffNeutralizedStrategy(
    fdr_target=0.10,  # 10% False Discovery Rate
    risk_aversion=1.0,
    max_leverage=2.0,
    random_state=42
)

strategy.fit(returns, risk_factors, alpha_factors)

# Get results
weights = strategy.get_portfolio_weights()
selected_signals = strategy.get_selected_signals()
factor_exposures = strategy.get_factor_exposures()

print(strategy.summary())
```

## Running Examples

### Basic Example (Synthetic Data)

```bash
cd examples
python basic_example.py
```

This demonstrates:
- Signal selection from a factor zoo (5 true + 45 noise signals)
- Factor neutralization
- FDR control verification

### Backtest Example (Time Series)

```bash
cd examples
python backtest_example.py
```

This demonstrates:
- Panel data handling
- Periodic refitting and rebalancing
- Performance visualization

## Running Tests

```bash
# Install pytest if needed
pip install pytest

# Run tests
cd tests
python test_strategy.py

# Or use pytest directly
pytest test_strategy.py -v
```

## Key Components

### 1. DataPreparation

Handles data organization and validation:

```python
from knockoff_neutralized import DataPreparation

prep = DataPreparation()
data = prep.prepare_data(returns, risk_factors, alpha_factors)

# Access components
print(data.returns.shape)
print(data.risk_factor_names)
print(data.alpha_factor_names)
```

### 2. ConditionalKnockoffFilter

Implements signal selection with FDR control:

```python
from knockoff_neutralized import ConditionalKnockoffFilter

kf = ConditionalKnockoffFilter(fdr_target=0.10, random_state=42)
kf.fit(Y=returns, F=risk_factors, A=alpha_factors)

# Get results
indices, names, w_stats = kf.get_selected_features()
print(f"Selected {len(indices)} signals")
print(f"Selected names: {names}")

# Transform to selected features only
A_selected = kf.transform(alpha_factors)
```

### 3. PortfolioOptimizer

Constructs neutralized portfolios:

```python
from knockoff_neutralized import PortfolioOptimizer

optimizer = PortfolioOptimizer(
    risk_aversion=1.0,
    max_leverage=2.0,
    neutralize_factors=True
)

result = optimizer.optimize(
    alpha_scores=scores,
    factor_exposures=betas,
    covariance_matrix=cov_matrix
)

print(f"Status: {result.status}")
print(f"Leverage: {result.leverage}")
print(f"Factor exposures: {result.factor_exposures}")
```

## Understanding the Parameters

### Strategy Parameters

- **`fdr_target`** (float, default=0.10): Target False Discovery Rate for signal selection. Lower values are more conservative (fewer signals selected).

- **`risk_aversion`** (float, default=1.0): Risk penalty in portfolio optimization. Higher values = more risk averse = lower volatility portfolios.

- **`max_leverage`** (float, default=2.0): Maximum gross leverage (sum of absolute weights). E.g., 2.0 means 100% long + 100% short.

- **`max_position_size`** (float, default=0.10): Maximum absolute weight for any single position (10% by default).

- **`transaction_cost`** (float, default=0.001): Transaction cost per dollar traded (0.1% by default).

- **`neutralize_factors`** (bool, default=True): Whether to enforce factor neutrality constraints.

- **`random_state`** (int, optional): Random seed for reproducibility.

## Data Format Requirements

### Input Formats

All three data types (returns, risk_factors, alpha_factors) can be provided in multiple formats:

**1. NumPy arrays:**
```python
returns = np.array([...])  # shape: (n_assets,)
risk_factors = np.array([...])  # shape: (n_assets, n_risk_factors)
alpha_factors = np.array([...])  # shape: (n_assets, n_alpha_factors)
```

**2. Pandas DataFrames:**
```python
returns = pd.Series([...], index=asset_ids)
risk_factors = pd.DataFrame({
    'Market': [...],
    'Size': [...],
    'Value': [...]
}, index=asset_ids)
alpha_factors = pd.DataFrame({...}, index=asset_ids)
```

**3. Dictionaries (recommended for named factors):**
```python
risk_factors = {
    'Market': np.array([...]),
    'Size': np.array([...])
}
alpha_factors = {
    'Signal_1': np.array([...]),
    'Signal_2': np.array([...]),
    ...
}
```

### Data Requirements

- All arrays must have the same number of assets (rows)
- No NaN or Inf values
- Returns should have non-zero variance
- At least one risk factor and one alpha factor

## Common Workflows

### Workflow 1: One-Time Signal Selection

```python
# Fit once to select signals
strategy.fit(returns, risk_factors, alpha_factors)

# Get selected signals
selected = strategy.get_selected_signals()
print(selected)

# Use selected signals in production
selected_names = selected['signal_name'].tolist()
```

### Workflow 2: Periodic Refitting

```python
# Initial fit
strategy.fit(returns_t0, risk_factors_t0, alpha_factors_t0)

# Later, refit with new data
strategy.fit(returns_t1, risk_factors_t1, alpha_factors_t1)
```

### Workflow 3: Rebalancing Without Refitting

```python
# Fit once
strategy.fit(returns_train, risk_factors_train, alpha_factors_train)

# Rebalance multiple times without refitting signals
for t in trading_periods:
    strategy.rebalance(
        returns=dummy_returns,  # Not used
        risk_factors=risk_factors_t,
        alpha_factors=alpha_factors_t,
        refit_signals=False  # Keep same signals
    )
    
    weights_t = strategy.get_portfolio_weights()
    # Execute trades...
```

### Workflow 4: Time Series Backtesting

```python
from knockoff_neutralized import DataPreparation

prep = DataPreparation()
dataset = prep.create_time_series_dataset(
    returns_panel=returns_df,
    risk_factors_panel=risk_factors_dict,
    alpha_factors_panel=alpha_factors_dict,
    forward_periods=1
)

# Backtest
for i, data in enumerate(dataset):
    if i % 20 == 0:  # Refit every 20 periods
        strategy.fit(data.returns, data.risk_factors, data.alpha_factors)
    else:  # Just rebalance
        strategy.rebalance(
            data.returns,
            data.risk_factors,
            data.alpha_factors,
            refit_signals=False
        )
    
    weights = strategy.get_portfolio_weights()
    # Calculate returns...
```

## Troubleshooting

### No Signals Selected

If `n_alpha_factors_selected = 0`, try:
- Lowering `fdr_target` (e.g., 0.20 instead of 0.10)
- Increasing sample size
- Checking if your alpha factors are actually predictive
- Verifying data quality (no NaN, reasonable variance)

### Optimization Fails

If portfolio optimization returns zero weights:
- Check factor exposure matrix for singularity
- Reduce `max_leverage` constraint
- Increase `risk_aversion`
- Provide covariance matrix explicitly

### Poor Out-of-Sample Performance

- Reduce `fdr_target` to be more conservative
- Increase refitting frequency
- Add more risk factors to condition on
- Check for data snooping / look-ahead bias

## Performance Considerations

### Speed

- Knockoff filter: O(n × p²) where p = number of alpha factors
- Portfolio optimization: O(n³) where n = number of assets
- For large problems, consider:
  - Reducing number of candidate factors
  - Using sparse covariance matrices
  - Running in parallel for multiple time periods

### Memory

- Memory scales with n_assets × n_factors
- For very large problems (1000+ assets), consider:
  - Batch processing
  - Sparse matrix representations
  - Chunking time series data

## References

### Academic Papers

1. **Knockoff Filters**:
   - Barber & Candès (2015): "Controlling the false discovery rate via knockoffs"
   - Candès et al. (2018): "Panning for gold: Model-X knockoffs"

2. **Factor Zoo Problem**:
   - Harvey et al. (2016): "...and the cross-section of expected returns"
   - McLean & Pontiff (2016): "Does academic research destroy stock return predictability?"

3. **Portfolio Construction**:
   - Grinold & Kahn (2000): "Active Portfolio Management"
   - Clarke et al. (2002): "Portfolio constraints and the fundamental law"

## Support

For issues or questions:
1. Check this guide
2. Review the examples
3. Examine the source code docstrings
4. Run the tests to verify installation

## License

MIT License - see LICENSE file for details
