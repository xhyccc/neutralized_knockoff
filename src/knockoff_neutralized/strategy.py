"""
Main Strategy Module

Orchestrates the complete knockoff-neutralized strategy:
1. Data preparation
2. Signal selection via conditional knockoffs
3. Portfolio construction with factor neutralization
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union
import warnings

from .data_preparation import DataPreparation, StrategyData
from .knockoff_filter import ConditionalKnockoffFilter
from .portfolio_optimizer import PortfolioOptimizer, OptimizationResult


class KnockoffNeutralizedStrategy:
    """
    Complete knockoff-neutralized quantitative strategy.
    
    This strategy combines:
    1. Conditional knockoff filters for robust signal selection
    2. Factor neutralization for portfolio construction
    3. Risk management through QP optimization
    
    Example
    -------
    >>> strategy = KnockoffNeutralizedStrategy(fdr_target=0.10)
    >>> strategy.fit(returns, risk_factors, alpha_factors)
    >>> weights = strategy.get_portfolio_weights()
    >>> selected_signals = strategy.get_selected_signals()
    """
    
    def __init__(
        self,
        fdr_target: float = 0.10,
        risk_aversion: float = 1.0,
        max_leverage: float = 2.0,
        max_position_size: float = 0.10,
        transaction_cost: float = 0.001,
        neutralize_factors: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Initialize the strategy.
        
        Parameters
        ----------
        fdr_target : float
            Target false discovery rate for knockoff filter (e.g., 0.10)
        risk_aversion : float
            Risk aversion parameter for portfolio optimization
        max_leverage : float
            Maximum gross leverage (sum of absolute weights)
        max_position_size : float
            Maximum absolute weight for any single position
        transaction_cost : float
            Transaction cost per dollar traded
        neutralize_factors : bool
            Whether to enforce factor neutrality constraints
        random_state : int, optional
            Random seed for reproducibility
        """
        self.fdr_target = fdr_target
        self.risk_aversion = risk_aversion
        self.max_leverage = max_leverage
        self.max_position_size = max_position_size
        self.transaction_cost = transaction_cost
        self.neutralize_factors = neutralize_factors
        self.random_state = random_state
        
        # Components
        self.data_prep = DataPreparation()
        self.knockoff_filter = ConditionalKnockoffFilter(
            fdr_target=fdr_target,
            random_state=random_state
        )
        self.portfolio_optimizer = PortfolioOptimizer(
            risk_aversion=risk_aversion,
            max_leverage=max_leverage,
            max_position_size=max_position_size,
            transaction_cost=transaction_cost,
            neutralize_factors=neutralize_factors
        )
        
        # State
        self.is_fitted_ = False
        self.current_data_ = None
        self.selected_alpha_indices_ = None
        self.selected_alpha_names_ = None
        self.current_weights_ = None
        self.optimization_result_ = None
        
    def fit(
        self,
        returns: Union[pd.DataFrame, np.ndarray],
        risk_factors: Union[pd.DataFrame, Dict[str, pd.DataFrame], np.ndarray],
        alpha_factors: Union[pd.DataFrame, Dict[str, pd.DataFrame], np.ndarray],
        asset_ids: Optional[List[str]] = None,
        covariance_matrix: Optional[np.ndarray] = None
    ) -> 'KnockoffNeutralizedStrategy':
        """
        Fit the strategy: select signals and optimize portfolio.
        
        Parameters
        ----------
        returns : DataFrame or ndarray
            Forward returns for each asset
        risk_factors : DataFrame, dict, or ndarray
            Known risk factors to neutralize against
        alpha_factors : DataFrame, dict, or ndarray
            Candidate alpha factors (the factor zoo)
        asset_ids : list of str, optional
            Asset identifiers
        covariance_matrix : ndarray, optional
            Asset return covariance matrix
            
        Returns
        -------
        self : KnockoffNeutralizedStrategy
            Fitted strategy
        """
        # Phase 1: Prepare data
        self.current_data_ = self.data_prep.prepare_data(
            returns=returns,
            risk_factors=risk_factors,
            alpha_factors=alpha_factors,
            asset_ids=asset_ids
        )
        
        # Phase 2: Signal selection via conditional knockoffs
        print(f"Phase 2: Running conditional knockoff filter...")
        print(f"  - Testing {self.current_data_.alpha_factors.shape[1]} alpha factors")
        print(f"  - Conditioning on {self.current_data_.risk_factors.shape[1]} risk factors")
        print(f"  - Target FDR: {self.fdr_target}")
        
        self.knockoff_filter.fit(
            Y=self.current_data_.returns,
            F=self.current_data_.risk_factors,
            A=self.current_data_.alpha_factors,
            alpha_factor_names=self.current_data_.alpha_factor_names
        )
        
        self.selected_alpha_indices_ = self.knockoff_filter.selected_indices_
        self.selected_alpha_names_ = self.knockoff_filter.selected_names_
        
        n_selected = len(self.selected_alpha_indices_)
        print(f"  ✓ Selected {n_selected} signals with FDR control")
        
        if n_selected == 0:
            warnings.warn(
                "No signals passed the knockoff filter. "
                "Returning zero weights."
            )
            self.current_weights_ = np.zeros(len(self.current_data_.returns))
            self.optimization_result_ = None
            self.is_fitted_ = True
            return self
        
        print(f"  - Selected signals: {self.selected_alpha_names_}")
        
        # Phase 3: Compute alpha scores from selected signals
        alpha_scores = self._compute_alpha_scores()
        
        # Phase 4: Portfolio construction with factor neutralization
        print(f"\nPhase 3: Constructing neutralized portfolio...")
        
        self.optimization_result_ = self.portfolio_optimizer.optimize(
            alpha_scores=alpha_scores,
            factor_exposures=self.current_data_.risk_factors,
            covariance_matrix=covariance_matrix,
            current_weights=self.current_weights_,
            factor_names=self.current_data_.risk_factor_names
        )
        
        self.current_weights_ = self.optimization_result_.weights
        
        print(f"  ✓ Optimization status: {self.optimization_result_.status}")
        print(f"  - Portfolio alpha score: {self.optimization_result_.alpha_score:.4f}")
        print(f"  - Portfolio variance: {self.optimization_result_.portfolio_variance:.6f}")
        print(f"  - Leverage: {self.optimization_result_.leverage:.2f}")
        
        if self.neutralize_factors:
            print(f"  - Factor exposures:")
            for name, exp in self.optimization_result_.factor_exposures.items():
                print(f"    {name}: {exp:.6f}")
        
        self.is_fitted_ = True
        return self
    
    def _compute_alpha_scores(self) -> np.ndarray:
        """
        Compute alpha scores from selected signals.
        
        Uses equal weighting by default, with sign from knockoff coefficients.
        """
        if len(self.selected_alpha_indices_) == 0:
            return np.zeros(len(self.current_data_.returns))
        
        # Get selected alpha factors
        selected_alphas = self.current_data_.alpha_factors[:, self.selected_alpha_indices_]
        
        # Get knockoff statistics (as proxy for signal strength)
        w_stats = self.knockoff_filter.w_statistics_[self.selected_alpha_indices_]
        
        # Normalize statistics to use as weights
        weights = w_stats / np.sum(np.abs(w_stats))
        
        # Compute weighted alpha score
        alpha_scores = selected_alphas @ weights
        
        # Standardize to have reasonable scale
        alpha_scores = (alpha_scores - np.mean(alpha_scores)) / (np.std(alpha_scores) + 1e-8)
        
        return alpha_scores
    
    def get_portfolio_weights(self) -> pd.Series:
        """
        Get current portfolio weights.
        
        Returns
        -------
        weights : pd.Series
            Portfolio weights indexed by asset IDs
        """
        if not self.is_fitted_:
            raise RuntimeError("Must call fit() before getting weights")
        
        return pd.Series(
            self.current_weights_,
            index=self.current_data_.asset_ids,
            name='portfolio_weights'
        )
    
    def get_selected_signals(self) -> pd.DataFrame:
        """
        Get information about selected signals.
        
        Returns
        -------
        signals_df : pd.DataFrame
            DataFrame with selected signal names and their knockoff statistics
        """
        if not self.is_fitted_:
            raise RuntimeError("Must call fit() before getting selected signals")
        
        if len(self.selected_alpha_indices_) == 0:
            return pd.DataFrame(columns=['signal_name', 'w_statistic'])
        
        w_stats = self.knockoff_filter.w_statistics_[self.selected_alpha_indices_]
        
        df = pd.DataFrame({
            'signal_name': self.selected_alpha_names_,
            'w_statistic': w_stats
        })
        
        return df.sort_values('w_statistic', ascending=False)
    
    def get_factor_exposures(self) -> pd.Series:
        """
        Get current factor exposures of the portfolio.
        
        Returns
        -------
        exposures : pd.Series
            Factor exposures indexed by factor names
        """
        if not self.is_fitted_:
            raise RuntimeError("Must call fit() before getting factor exposures")
        
        if self.optimization_result_ is None:
            return pd.Series(dtype=float)
        
        return pd.Series(
            self.optimization_result_.factor_exposures,
            name='factor_exposures'
        )
    
    def predict_returns(
        self,
        risk_factors: Union[pd.DataFrame, Dict[str, pd.DataFrame], np.ndarray],
        alpha_factors: Union[pd.DataFrame, Dict[str, pd.DataFrame], np.ndarray],
        asset_ids: Optional[List[str]] = None
    ) -> pd.Series:
        """
        Predict returns for new data using the fitted model.
        
        Parameters
        ----------
        risk_factors : DataFrame, dict, or ndarray
            Risk factors for new data
        alpha_factors : DataFrame, dict, or ndarray
            Alpha factors for new data
        asset_ids : list of str, optional
            Asset identifiers
            
        Returns
        -------
        predictions : pd.Series
            Predicted returns (alpha scores)
        """
        if not self.is_fitted_:
            raise RuntimeError("Must call fit() before predict_returns()")
        
        # Prepare new data
        # Create dummy returns (not used for prediction)
        dummy_returns = np.zeros(
            alpha_factors.shape[0] if isinstance(alpha_factors, np.ndarray)
            else len(next(iter(alpha_factors.values())))
        )
        
        new_data = self.data_prep.prepare_data(
            returns=dummy_returns,
            risk_factors=risk_factors,
            alpha_factors=alpha_factors,
            asset_ids=asset_ids
        )
        
        # Compute alpha scores using selected factors
        if len(self.selected_alpha_indices_) == 0:
            predictions = np.zeros(len(new_data.returns))
        else:
            selected_alphas = new_data.alpha_factors[:, self.selected_alpha_indices_]
            w_stats = self.knockoff_filter.w_statistics_[self.selected_alpha_indices_]
            weights = w_stats / np.sum(np.abs(w_stats))
            predictions = selected_alphas @ weights
        
        return pd.Series(
            predictions,
            index=new_data.asset_ids,
            name='predicted_returns'
        )
    
    def rebalance(
        self,
        returns: Union[pd.DataFrame, np.ndarray],
        risk_factors: Union[pd.DataFrame, Dict[str, pd.DataFrame], np.ndarray],
        alpha_factors: Union[pd.DataFrame, Dict[str, pd.DataFrame], np.ndarray],
        asset_ids: Optional[List[str]] = None,
        covariance_matrix: Optional[np.ndarray] = None,
        refit_signals: bool = False
    ) -> 'KnockoffNeutralizedStrategy':
        """
        Rebalance the portfolio with new data.
        
        Parameters
        ----------
        returns : DataFrame or ndarray
            New forward returns
        risk_factors : DataFrame, dict, or ndarray
            New risk factors
        alpha_factors : DataFrame, dict, or ndarray
            New alpha factors
        asset_ids : list of str, optional
            Asset identifiers
        covariance_matrix : ndarray, optional
            New covariance matrix
        refit_signals : bool
            If True, re-run knockoff filter. If False, use existing selection.
            
        Returns
        -------
        self : KnockoffNeutralizedStrategy
            Rebalanced strategy
        """
        if refit_signals or not self.is_fitted_:
            # Full refit
            return self.fit(
                returns=returns,
                risk_factors=risk_factors,
                alpha_factors=alpha_factors,
                asset_ids=asset_ids,
                covariance_matrix=covariance_matrix
            )
        
        # Just reoptimize with existing signal selection
        self.current_data_ = self.data_prep.prepare_data(
            returns=returns,
            risk_factors=risk_factors,
            alpha_factors=alpha_factors,
            asset_ids=asset_ids
        )
        
        alpha_scores = self._compute_alpha_scores()
        
        self.optimization_result_ = self.portfolio_optimizer.optimize(
            alpha_scores=alpha_scores,
            factor_exposures=self.current_data_.risk_factors,
            covariance_matrix=covariance_matrix,
            current_weights=self.current_weights_,
            factor_names=self.current_data_.risk_factor_names
        )
        
        self.current_weights_ = self.optimization_result_.weights
        
        return self
    
    def summary(self) -> Dict:
        """
        Get a summary of the current strategy state.
        
        Returns
        -------
        summary : dict
            Dictionary with strategy statistics
        """
        if not self.is_fitted_:
            return {'status': 'not_fitted'}
        
        summary = {
            'status': 'fitted',
            'n_assets': len(self.current_data_.asset_ids),
            'n_risk_factors': len(self.current_data_.risk_factor_names),
            'n_alpha_factors_tested': len(self.current_data_.alpha_factor_names),
            'n_alpha_factors_selected': len(self.selected_alpha_indices_),
            'selected_signals': self.selected_alpha_names_,
            'fdr_target': self.fdr_target,
            'portfolio_leverage': self.optimization_result_.leverage if self.optimization_result_ else 0.0,
            'portfolio_alpha_score': self.optimization_result_.alpha_score if self.optimization_result_ else 0.0,
            'factor_exposures': self.optimization_result_.factor_exposures if self.optimization_result_ else {},
        }
        
        return summary
    
    def __repr__(self) -> str:
        """String representation of the strategy."""
        if not self.is_fitted_:
            return f"KnockoffNeutralizedStrategy(fdr_target={self.fdr_target}, not fitted)"
        
        return (
            f"KnockoffNeutralizedStrategy(\n"
            f"  fdr_target={self.fdr_target},\n"
            f"  n_selected_signals={len(self.selected_alpha_indices_)},\n"
            f"  leverage={self.optimization_result_.leverage:.2f},\n"
            f"  alpha_score={self.optimization_result_.alpha_score:.4f}\n"
            f")"
        )
