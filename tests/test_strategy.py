"""
Unit Tests for Knockoff-Neutralized Strategy
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from knockoff_neutralized import (
    KnockoffNeutralizedStrategy,
    DataPreparation,
    ConditionalKnockoffFilter,
    PortfolioOptimizer
)


class TestDataPreparation:
    """Test data preparation module"""
    
    def test_prepare_data_basic(self):
        """Test basic data preparation"""
        prep = DataPreparation()
        
        n_assets = 50
        returns = np.random.randn(n_assets)
        risk_factors = np.random.randn(n_assets, 3)
        alpha_factors = np.random.randn(n_assets, 10)
        
        data = prep.prepare_data(returns, risk_factors, alpha_factors)
        
        assert data.returns.shape == (n_assets,)
        assert data.risk_factors.shape == (n_assets, 3)
        assert data.alpha_factors.shape == (n_assets, 10)
        assert len(data.asset_ids) == n_assets
    
    def test_prepare_data_with_dict(self):
        """Test data preparation with dictionary inputs"""
        prep = DataPreparation()
        
        n_assets = 50
        returns = np.random.randn(n_assets)
        risk_factors = {
            'Market': np.random.randn(n_assets),
            'Size': np.random.randn(n_assets)
        }
        alpha_factors = {
            'Alpha1': np.random.randn(n_assets),
            'Alpha2': np.random.randn(n_assets)
        }
        
        data = prep.prepare_data(returns, risk_factors, alpha_factors)
        
        assert data.risk_factors.shape == (n_assets, 2)
        assert data.alpha_factors.shape == (n_assets, 2)
        assert 'Market' in data.risk_factor_names
        assert 'Alpha1' in data.alpha_factor_names
    
    def test_standardize_factors(self):
        """Test factor standardization"""
        prep = DataPreparation()
        
        factors = np.random.randn(100, 5) * 3 + 10
        standardized, mean, std = prep.standardize_factors(factors)
        
        assert np.allclose(np.mean(standardized, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(standardized, axis=0), 1, atol=1e-10)


class TestConditionalKnockoffFilter:
    """Test knockoff filter module"""
    
    def test_fit_basic(self):
        """Test basic knockoff filter fitting"""
        np.random.seed(42)
        
        n_samples = 200
        n_risk = 3
        n_alpha = 10
        
        F = np.random.randn(n_samples, n_risk)
        A = np.random.randn(n_samples, n_alpha)
        
        # Create true signal in first alpha
        Y = A[:, 0] * 0.5 + F @ np.array([0.3, -0.2, 0.1]) + np.random.randn(n_samples) * 0.5
        
        kf = ConditionalKnockoffFilter(fdr_target=0.10, random_state=42)
        kf.fit(Y, F, A)
        
        assert kf.selected_indices_ is not None
        assert kf.w_statistics_ is not None
        assert len(kf.w_statistics_) == n_alpha
    
    def test_transform(self):
        """Test feature transformation"""
        np.random.seed(42)
        
        n_samples = 200
        F = np.random.randn(n_samples, 2)
        A = np.random.randn(n_samples, 10)
        Y = A[:, 0] + np.random.randn(n_samples) * 0.5
        
        kf = ConditionalKnockoffFilter(fdr_target=0.20, random_state=42)
        kf.fit(Y, F, A)
        
        A_selected = kf.transform(A)
        
        assert A_selected.shape[0] == n_samples
        assert A_selected.shape[1] == len(kf.selected_indices_)


class TestPortfolioOptimizer:
    """Test portfolio optimizer module"""
    
    def test_optimize_basic(self):
        """Test basic portfolio optimization"""
        optimizer = PortfolioOptimizer(
            risk_aversion=1.0,
            max_leverage=2.0,
            neutralize_factors=True
        )
        
        n_assets = 50
        alpha_scores = np.random.randn(n_assets)
        factor_exposures = np.random.randn(n_assets, 3)
        
        result = optimizer.optimize(
            alpha_scores=alpha_scores,
            factor_exposures=factor_exposures
        )
        
        assert result.weights.shape == (n_assets,)
        assert result.success
        assert np.sum(result.weights) < 1e-4  # Dollar neutral
        assert np.sum(np.abs(result.weights)) <= 2.0 + 1e-4  # Leverage constraint
    
    def test_factor_neutrality(self):
        """Test that portfolio is factor neutral"""
        optimizer = PortfolioOptimizer(
            risk_aversion=1.0,
            neutralize_factors=True
        )
        
        n_assets = 100
        alpha_scores = np.random.randn(n_assets)
        factor_exposures = np.random.randn(n_assets, 2)
        
        result = optimizer.optimize(
            alpha_scores=alpha_scores,
            factor_exposures=factor_exposures
        )
        
        # Check factor exposures are near zero
        for exposure in result.factor_exposures.values():
            assert abs(exposure) < 1e-3


class TestKnockoffNeutralizedStrategy:
    """Test main strategy class"""
    
    def test_fit_basic(self):
        """Test basic strategy fitting"""
        np.random.seed(42)
        
        n_assets = 100
        returns = np.random.randn(n_assets)
        risk_factors = {
            'Market': np.random.randn(n_assets),
            'Size': np.random.randn(n_assets)
        }
        alpha_factors = {
            f'Alpha{i}': np.random.randn(n_assets) for i in range(20)
        }
        
        strategy = KnockoffNeutralizedStrategy(
            fdr_target=0.15,
            random_state=42
        )
        
        strategy.fit(returns, risk_factors, alpha_factors)
        
        assert strategy.is_fitted_
        assert strategy.current_weights_ is not None
        assert len(strategy.current_weights_) == n_assets
    
    def test_get_portfolio_weights(self):
        """Test getting portfolio weights"""
        np.random.seed(42)
        
        n_assets = 50
        returns = np.random.randn(n_assets)
        risk_factors = np.random.randn(n_assets, 2)
        alpha_factors = np.random.randn(n_assets, 10)
        
        strategy = KnockoffNeutralizedStrategy(random_state=42)
        strategy.fit(returns, risk_factors, alpha_factors)
        
        weights = strategy.get_portfolio_weights()
        
        assert len(weights) == n_assets
        assert abs(weights.sum()) < 1e-4  # Dollar neutral
    
    def test_get_selected_signals(self):
        """Test getting selected signals"""
        np.random.seed(42)
        
        n_assets = 100
        returns = np.random.randn(n_assets)
        risk_factors = np.random.randn(n_assets, 2)
        alpha_factors = np.random.randn(n_assets, 15)
        
        alpha_names = [f'Alpha_{i}' for i in range(15)]
        
        strategy = KnockoffNeutralizedStrategy(fdr_target=0.20, random_state=42)
        
        # Create dict format
        alpha_dict = {name: alpha_factors[:, i] for i, name in enumerate(alpha_names)}
        
        strategy.fit(returns, risk_factors, alpha_dict)
        
        signals = strategy.get_selected_signals()
        
        assert 'signal_name' in signals.columns
        assert 'w_statistic' in signals.columns
    
    def test_summary(self):
        """Test strategy summary"""
        np.random.seed(42)
        
        n_assets = 50
        returns = np.random.randn(n_assets)
        risk_factors = np.random.randn(n_assets, 2)
        alpha_factors = np.random.randn(n_assets, 10)
        
        strategy = KnockoffNeutralizedStrategy(random_state=42)
        strategy.fit(returns, risk_factors, alpha_factors)
        
        summary = strategy.summary()
        
        assert summary['status'] == 'fitted'
        assert 'n_assets' in summary
        assert 'n_alpha_factors_selected' in summary


def run_tests():
    """Run all tests"""
    print("Running tests...")
    pytest.main([__file__, '-v'])


if __name__ == "__main__":
    run_tests()
