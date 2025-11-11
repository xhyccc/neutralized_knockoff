"""
Knockoff-Neutralized Quantitative Strategy

A robust quantitative trading strategy that combines conditional knockoff filters
for signal selection with factor neutralization for portfolio construction.
"""

from .strategy import KnockoffNeutralizedStrategy
from .data_preparation import DataPreparation
from .knockoff_filter import ConditionalKnockoffFilter
from .portfolio_optimizer import PortfolioOptimizer

__version__ = "0.1.0"
__all__ = [
    "KnockoffNeutralizedStrategy",
    "DataPreparation",
    "ConditionalKnockoffFilter",
    "PortfolioOptimizer",
]
