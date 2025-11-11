"""
Portfolio Optimizer Module

Implements quadratic programming-based portfolio construction with:
- Factor neutralization constraints
- Position and leverage limits
- Risk minimization
- Transaction cost awareness
"""

import numpy as np
import cvxpy as cp
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class OptimizationResult:
    """Container for portfolio optimization results"""
    weights: np.ndarray
    alpha_score: float
    portfolio_variance: float
    leverage: float
    turnover: float
    factor_exposures: Dict[str, float]
    status: str
    success: bool


class PortfolioOptimizer:
    """
    Constructs optimal portfolios with factor neutralization.
    
    Solves a quadratic program to maximize alpha exposure while:
    - Neutralizing exposures to known risk factors
    - Minimizing portfolio variance
    - Respecting position and leverage constraints
    """
    
    def __init__(
        self,
        risk_aversion: float = 1.0,
        max_leverage: float = 2.0,
        max_position_size: float = 0.10,
        max_sector_exposure: Optional[float] = None,
        transaction_cost: float = 0.001,
        neutralize_factors: bool = True,
        long_only: bool = False,
        beta_neutral: bool = True,
        target_beta: Optional[float] = None,
        beta_tolerance: float = 0.15
    ):
        """
        Initialize the portfolio optimizer.
        
        Parameters
        ----------
        risk_aversion : float
            Risk aversion parameter (higher = more risk averse)
        max_leverage : float
            Maximum gross leverage (sum of absolute weights)
        max_position_size : float
            Maximum absolute weight for any single position
        max_sector_exposure : float, optional
            Maximum absolute sector exposure
        transaction_cost : float
            Transaction cost per dollar traded
        neutralize_factors : bool
            Whether to enforce factor neutrality constraints
        beta_neutral : bool
            If True, enforce absolute beta=0 for the first risk factor (assumed to be market beta).
            Ignored if target_beta is specified.
        target_beta : float, optional
            Target beta for the portfolio (e.g., 1.0 for market-like exposure).
            If specified, overrides beta_neutral and enforces beta ≈ target_beta.
            Useful for long-only portfolios where absolute beta=0 is infeasible.
        beta_tolerance : float
            Tolerance for target_beta constraint: |portfolio_beta - target_beta| <= beta_tolerance.
            Only used when target_beta is specified. Default: 0.15
        """
        self.risk_aversion = risk_aversion
        self.max_leverage = max_leverage
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.transaction_cost = transaction_cost
        self.neutralize_factors = neutralize_factors
        self.long_only = long_only
        self.beta_neutral = beta_neutral
        self.target_beta = target_beta
        self.beta_tolerance = beta_tolerance
        
        # State
        self.current_weights_ = None
        self.factor_betas_ = None
        self.covariance_matrix_ = None
        
    def optimize(
        self,
        alpha_scores: np.ndarray,
        factor_exposures: np.ndarray,
        covariance_matrix: Optional[np.ndarray] = None,
        current_weights: Optional[np.ndarray] = None,
        sector_memberships: Optional[np.ndarray] = None,
        factor_names: Optional[List[str]] = None
    ) -> OptimizationResult:
        """
        Optimize portfolio weights.
        
        Parameters
        ----------
        alpha_scores : ndarray, shape (n_assets,)
            Alpha scores for each asset (higher = more attractive)
        factor_exposures : ndarray, shape (n_assets, n_factors)
            Factor loadings (betas) for each asset
        covariance_matrix : ndarray, shape (n_assets, n_assets), optional
            Asset return covariance matrix (if None, uses diagonal)
        current_weights : ndarray, shape (n_assets,), optional
            Current portfolio weights for turnover calculation
        sector_memberships : ndarray, shape (n_assets, n_sectors), optional
            One-hot encoded sector memberships
        factor_names : list of str, optional
            Names of factors for reporting
            
        Returns
        -------
        OptimizationResult
            Optimization results including weights and diagnostics
        """
        alpha_scores = np.asarray(alpha_scores, dtype=float)
        factor_exposures = np.asarray(factor_exposures, dtype=float)
        if factor_exposures.ndim == 1:
            factor_exposures = factor_exposures.reshape(-1, 1)

        n_assets = len(alpha_scores)
        n_factors = factor_exposures.shape[1]
        
        if current_weights is None:
            current_weights = np.zeros(n_assets)
        
        if covariance_matrix is None:
            # Use diagonal covariance (no cross-asset correlation)
            covariance_matrix = np.eye(n_assets) * 0.01
        else:
            covariance_matrix = np.asarray(covariance_matrix, dtype=float)
        
        if factor_names is None:
            factor_names = [f"factor_{i}" for i in range(n_factors)]
        
        # Define optimization variables
        w = cp.Variable(n_assets)
        
        # Objective: maximize alpha - risk penalty - transaction costs
        alpha_term = alpha_scores @ w
        risk_term = cp.quad_form(w, covariance_matrix)
        turnover = cp.sum(cp.abs(w - current_weights))
        tc_term = self.transaction_cost * turnover
        
        objective = cp.Maximize(alpha_term - self.risk_aversion * risk_term - tc_term)
        
        # Constraints
        constraints = []
        
        # 1. Gross leverage constraint
        constraints.append(cp.sum(cp.abs(w)) <= self.max_leverage)

        # 2. Budget and position limits
        if self.long_only:
            constraints.append(cp.sum(w) == 1)
            constraints.append(w >= 0)
            constraints.append(w <= self.max_position_size)
        else:
            constraints.append(cp.sum(w) == 0)
            constraints.append(w <= self.max_position_size)
            constraints.append(w >= -self.max_position_size)
        
        # 4. Factor neutrality or target beta
        if self.target_beta is not None and n_factors > 0:
            # Target beta mode (for long-only portfolios)
            # Constrain portfolio beta to be near target_beta (e.g., 1.0 for market-like)
            beta_exposure = factor_exposures[:, 0] @ w
            constraints.append(beta_exposure <= self.target_beta + self.beta_tolerance)
            constraints.append(beta_exposure >= self.target_beta - self.beta_tolerance)
            
            # Still neutralize other factors (sectors, etc.)
            for i in range(1, n_factors):
                if self.long_only:
                    centered_factor = factor_exposures[:, i] - factor_exposures[:, i].mean()
                    factor_exposure_i = centered_factor @ w
                else:
                    factor_exposure_i = factor_exposures[:, i] @ w
                constraints.append(factor_exposure_i <= 1e-6)
                constraints.append(factor_exposure_i >= -1e-6)
                
        elif self.neutralize_factors and n_factors > 0:
            # Original factor neutrality mode
            for i in range(n_factors):
                # First factor is assumed to be market beta; enforce absolute neutrality if requested
                if i == 0 and self.beta_neutral:
                    beta_exposure = factor_exposures[:, i] @ w
                    constraints.append(beta_exposure <= 1e-6)
                    constraints.append(beta_exposure >= -1e-6)
                else:
                    # For other factors (sectors), enforce benchmark-relative neutrality in long-only mode
                    if self.long_only:
                        # Center factor exposure relative to equal-weight benchmark
                        centered_factor = factor_exposures[:, i] - factor_exposures[:, i].mean()
                        factor_exposure_i = centered_factor @ w
                    else:
                        factor_exposure_i = factor_exposures[:, i] @ w
                    
                    constraints.append(factor_exposure_i <= 1e-6)
                    constraints.append(factor_exposure_i >= -1e-6)
        
        # 5. Sector constraints (if provided)
        if sector_memberships is not None and self.max_sector_exposure is not None:
            n_sectors = sector_memberships.shape[1]
            for s in range(n_sectors):
                sector_exposure = sector_memberships[:, s] @ w
                constraints.append(sector_exposure <= self.max_sector_exposure)
                constraints.append(sector_exposure >= -self.max_sector_exposure)
        
        # Solve the problem
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status in ['optimal', 'optimal_inaccurate']:
                weights = w.value
                success = True
                status = problem.status
            else:
                # Fallback: return zero weights
                weights = np.zeros(n_assets)
                success = False
                status = problem.status
        except Exception as e:
            weights = np.zeros(n_assets)
            success = False
            status = f"error: {str(e)}"
        
        if success and self.long_only:
            weights = np.maximum(weights, 0.0)
            total_w = weights.sum()
            if total_w > 1e-12:
                weights = weights / total_w
            else:
                weights = np.zeros(n_assets)
                success = False
                status = "infeasible_long_only"

        # Compute diagnostics
        alpha_score = np.dot(alpha_scores, weights)
        portfolio_variance = weights @ covariance_matrix @ weights
        leverage = np.sum(np.abs(weights))
        turnover_val = np.sum(np.abs(weights - current_weights))
        
        # Factor exposures
        factor_exp_dict = {}
        raw_factor_exposures = np.asarray(factor_exposures, dtype=float)
        for i, name in enumerate(factor_names):
            factor_exp_dict[name] = np.dot(raw_factor_exposures[:, i], weights)
        
        return OptimizationResult(
            weights=weights,
            alpha_score=alpha_score,
            portfolio_variance=portfolio_variance,
            leverage=leverage,
            turnover=turnover_val,
            factor_exposures=factor_exp_dict,
            status=status,
            success=success
        )
    
    def estimate_factor_betas(
        self,
        returns_history: np.ndarray,
        factor_returns_history: np.ndarray,
        window: Optional[int] = None
    ) -> np.ndarray:
        """
        Estimate factor betas (loadings) for each asset.
        
        Uses rolling window regression: r_i,t = α_i + Σ β_i,j * f_j,t + ε_i,t
        
        Parameters
        ----------
        returns_history : ndarray, shape (n_periods, n_assets)
            Historical asset returns
        factor_returns_history : ndarray, shape (n_periods, n_factors)
            Historical factor returns
        window : int, optional
            Rolling window length (None = use all data)
            
        Returns
        -------
        betas : ndarray, shape (n_assets, n_factors)
            Estimated factor betas
        """
        n_periods, n_assets = returns_history.shape
        n_factors = factor_returns_history.shape[1]
        
        if window is None:
            window = n_periods
        
        # Use most recent window
        start_idx = max(0, n_periods - window)
        returns_window = returns_history[start_idx:]
        factors_window = factor_returns_history[start_idx:]
        
        # Run regression for each asset
        betas = np.zeros((n_assets, n_factors))
        
        for i in range(n_assets):
            y = returns_window[:, i]
            X = factors_window
            
            # OLS: β = (X'X)^(-1) X'y
            # Add small ridge for stability
            XtX = X.T @ X + 1e-6 * np.eye(n_factors)
            Xty = X.T @ y
            betas[i] = np.linalg.solve(XtX, Xty)
        
        self.factor_betas_ = betas
        return betas
    
    def estimate_covariance(
        self,
        returns_history: np.ndarray,
        method: str = 'sample',
        shrinkage: float = 0.1,
        window: Optional[int] = None
    ) -> np.ndarray:
        """
        Estimate asset return covariance matrix.
        
        Parameters
        ----------
        returns_history : ndarray, shape (n_periods, n_assets)
            Historical asset returns
        method : str
            Covariance estimation method: 'sample', 'shrinkage', or 'diagonal'
        shrinkage : float
            Shrinkage intensity for 'shrinkage' method (0 = no shrinkage)
        window : int, optional
            Rolling window length (None = use all data)
            
        Returns
        -------
        cov_matrix : ndarray, shape (n_assets, n_assets)
            Estimated covariance matrix
        """
        n_periods, n_assets = returns_history.shape
        
        if window is None:
            window = n_periods
        
        # Use most recent window
        start_idx = max(0, n_periods - window)
        returns_window = returns_history[start_idx:]
        
        # Center returns
        returns_centered = returns_window - np.mean(returns_window, axis=0)
        
        if method == 'sample':
            # Sample covariance
            cov_matrix = (returns_centered.T @ returns_centered) / len(returns_window)
        
        elif method == 'shrinkage':
            # Ledoit-Wolf shrinkage toward diagonal
            sample_cov = (returns_centered.T @ returns_centered) / len(returns_window)
            
            # Target: diagonal matrix
            variances = np.diag(sample_cov)
            target = np.diag(variances)
            
            # Shrink
            cov_matrix = (1 - shrinkage) * sample_cov + shrinkage * target
        
        elif method == 'diagonal':
            # Diagonal covariance only
            variances = np.var(returns_window, axis=0)
            cov_matrix = np.diag(variances)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Add small ridge for numerical stability
        cov_matrix += 1e-6 * np.eye(n_assets)
        
        self.covariance_matrix_ = cov_matrix
        return cov_matrix
    
    def backtest_portfolio(
        self,
        alpha_scores_history: np.ndarray,
        returns_history: np.ndarray,
        factor_exposures_history: np.ndarray,
        rebalance_frequency: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Backtest the portfolio optimization strategy.
        
        Parameters
        ----------
        alpha_scores_history : ndarray, shape (n_periods, n_assets)
            Historical alpha scores
        returns_history : ndarray, shape (n_periods, n_assets)
            Historical returns (for computing realized performance)
        factor_exposures_history : ndarray, shape (n_periods, n_assets, n_factors)
            Historical factor exposures
        rebalance_frequency : int
            Rebalance every N periods
            
        Returns
        -------
        results : dict
            Backtest results with keys:
            - 'weights': Portfolio weights over time
            - 'returns': Portfolio returns
            - 'alpha_scores': Portfolio alpha scores
            - 'leverage': Portfolio leverage over time
        """
        n_periods, n_assets = alpha_scores_history.shape
        
        # Storage
        weights_history = np.zeros((n_periods, n_assets))
        portfolio_returns = np.zeros(n_periods)
        alpha_score_realized = np.zeros(n_periods)
        leverage_history = np.zeros(n_periods)
        
        current_weights = np.zeros(n_assets)
        
        for t in range(n_periods):
            # Rebalance if needed
            if t % rebalance_frequency == 0:
                # Optimize
                result = self.optimize(
                    alpha_scores=alpha_scores_history[t],
                    factor_exposures=factor_exposures_history[t],
                    current_weights=current_weights
                )
                
                if result.success:
                    current_weights = result.weights
            
            # Store weights
            weights_history[t] = current_weights
            
            # Compute realized return
            portfolio_returns[t] = np.dot(current_weights, returns_history[t])
            
            # Compute alpha score
            alpha_score_realized[t] = np.dot(current_weights, alpha_scores_history[t])
            
            # Compute leverage
            leverage_history[t] = np.sum(np.abs(current_weights))
        
        return {
            'weights': weights_history,
            'returns': portfolio_returns,
            'alpha_scores': alpha_score_realized,
            'leverage': leverage_history
        }
