# Knockoff-Neutralized Quantitative Strategy

A robust quantitative trading strategy that combines **Conditional Knockoff Filters** for signal selection with **Factor Neutralization** for portfolio construction.

## Overview

This strategy solves two critical problems in quantitative finance:

1. **The Factor Zoo Problem**: Using knockoff filters to identify which alpha signals are truly predictive vs. lucky noise
2. **Factor Risk Management**: Neutralizing exposures to common risk factors to isolate true alpha

## Key Features

- **Conditional Knockoff Filters**: Statistically rigorous signal selection with FDR control
- **Factor Neutralization**: Quadratic programming-based portfolio optimization
- **Modular Design**: Separate components for data prep, signal selection, and portfolio construction
- **Out-of-Sample Robustness**: Built to avoid overfitting and p-hacking

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from knockoff_neutralized import KnockoffNeutralizedStrategy
import numpy as np
import pandas as pd

# Prepare your data
returns = pd.DataFrame(...)  # Forward returns (N x T)
risk_factors = pd.DataFrame(...)  # Known risk factors (N x K x T)
alpha_factors = pd.DataFrame(...)  # Candidate alpha factors (N x P x T)

# Initialize strategy
strategy = KnockoffNeutralizedStrategy(
    fdr_target=0.10,  # 10% False Discovery Rate
    risk_aversion=1.0,
    max_leverage=2.0
)

# Run the strategy
strategy.fit(returns, risk_factors, alpha_factors)
weights = strategy.get_portfolio_weights()
```

## Project Structure

```
knockoff_neutralized/
├── src/
│   └── knockoff_neutralized/
│       ├── __init__.py
│       ├── data_preparation.py      # Data handling and preprocessing
│       ├── knockoff_filter.py       # Conditional knockoff generation
│       ├── portfolio_optimizer.py   # QP-based portfolio construction
│       └── strategy.py              # Main strategy orchestration
├── examples/
│   └── basic_example.py             # Example usage with synthetic data
├── tests/
│   └── test_strategy.py             # Unit tests
├── requirements.txt
├── setup.py
└── README.md
```

## Methodology

### Phase 1: Data Preparation
- Define target variable Y (forward returns)
- Specify known risk factors F (market, size, value, momentum, etc.)
- Collect candidate alpha factors A (your factor zoo)

### Phase 2: Signal Selection via Conditional Knockoffs
1. Model the conditional distribution A|F
2. Generate knockoff variables Ã
3. Run Lasso regression on [F, A, Ã]
4. Compute knockoff statistics W_j = |γ_j| - |γ̃_j|
5. Apply FDR control to select robust signals

### Phase 3: Portfolio Construction
1. Calculate alpha scores from selected signals
2. Solve quadratic program with factor neutrality constraints
3. Apply position and leverage limits
4. Execute trades

## References

- Barber, R. F., & Candès, E. J. (2015). "Controlling the false discovery rate via knockoffs"
- Candès, E., Fan, Y., Janson, L., & Lv, J. (2018). "Panning for gold: Model-X knockoffs for high dimensional controlled variable selection"
- Harvey, C. R., Liu, Y., & Zhu, H. (2016). "...and the cross-section of expected returns"

## License

MIT License
