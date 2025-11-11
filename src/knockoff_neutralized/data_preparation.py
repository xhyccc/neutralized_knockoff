"""
Data Preparation Module

Handles the organization and preprocessing of:
- Target variables (forward returns)
- Known risk factors (market, size, value, etc.)
- Candidate alpha factors (the factor zoo)
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class StrategyData:
    """Container for strategy data at a given time point"""
    returns: np.ndarray  # Shape: (n_assets,)
    risk_factors: np.ndarray  # Shape: (n_assets, n_risk_factors)
    alpha_factors: np.ndarray  # Shape: (n_assets, n_alpha_factors)
    asset_ids: List[str]
    risk_factor_names: List[str]
    alpha_factor_names: List[str]
    timestamp: Optional[pd.Timestamp] = None


class DataPreparation:
    """
    Prepares and validates data for the knockoff-neutralized strategy.
    
    Separates data into three categories:
    1. Target variable Y: Forward returns
    2. Known risk factors F: Factors to neutralize against
    3. Candidate alpha factors A: The factor zoo to test
    """
    
    def __init__(self):
        self.risk_factor_names = []
        self.alpha_factor_names = []
        self.asset_ids = []
        
    def prepare_data(
        self,
        returns: Union[pd.DataFrame, np.ndarray],
        risk_factors: Union[pd.DataFrame, Dict[str, pd.DataFrame], np.ndarray],
        alpha_factors: Union[pd.DataFrame, Dict[str, pd.DataFrame], np.ndarray],
        asset_ids: Optional[List[str]] = None,
        timestamp: Optional[pd.Timestamp] = None
    ) -> StrategyData:
        """
        Prepare and validate all data components.
        
        Parameters
        ----------
        returns : DataFrame or ndarray
            Forward returns for each asset. Shape: (n_assets,) or (n_assets, 1)
        risk_factors : DataFrame, dict of DataFrames, or ndarray
            Known risk factors. If dict, keys are factor names.
            Shape: (n_assets, n_risk_factors)
        alpha_factors : DataFrame, dict of DataFrames, or ndarray
            Candidate alpha factors. If dict, keys are factor names.
            Shape: (n_assets, n_alpha_factors)
        asset_ids : list, optional
            List of asset identifiers
        timestamp : pd.Timestamp, optional
            Timestamp for this data snapshot
            
        Returns
        -------
        StrategyData
            Validated and organized data
        """
        # Convert returns to array
        if isinstance(returns, pd.DataFrame):
            asset_ids = asset_ids or returns.index.tolist()
            returns_arr = returns.values.flatten()
        else:
            returns_arr = np.asarray(returns).flatten()
            asset_ids = asset_ids or [f"asset_{i}" for i in range(len(returns_arr))]
        
        n_assets = len(returns_arr)
        
        # Convert risk factors to array
        risk_factors_arr, risk_factor_names = self._convert_to_array(
            risk_factors, n_assets, "risk_factor"
        )
        
        # Convert alpha factors to array
        alpha_factors_arr, alpha_factor_names = self._convert_to_array(
            alpha_factors, n_assets, "alpha_factor"
        )
        
        # Validate shapes
        self._validate_data(returns_arr, risk_factors_arr, alpha_factors_arr)
        
        # Store names for later use
        self.asset_ids = asset_ids
        self.risk_factor_names = risk_factor_names
        self.alpha_factor_names = alpha_factor_names
        
        return StrategyData(
            returns=returns_arr,
            risk_factors=risk_factors_arr,
            alpha_factors=alpha_factors_arr,
            asset_ids=asset_ids,
            risk_factor_names=risk_factor_names,
            alpha_factor_names=alpha_factor_names,
            timestamp=timestamp
        )
    
    def _convert_to_array(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame], np.ndarray],
        n_assets: int,
        prefix: str
    ) -> Tuple[np.ndarray, List[str]]:
        """Convert various input formats to standardized array format."""
        if isinstance(data, dict):
            # Dictionary of factors
            factor_names = list(data.keys())
            arrays = []
            for name, factor_data in data.items():
                if isinstance(factor_data, pd.DataFrame):
                    arr = factor_data.values
                else:
                    arr = np.asarray(factor_data)
                
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                arrays.append(arr)
            
            result = np.hstack(arrays)
            
        elif isinstance(data, pd.DataFrame):
            result = data.values
            if result.ndim == 1:
                result = result.reshape(-1, 1)
            factor_names = data.columns.tolist()
            
        else:
            result = np.asarray(data)
            if result.ndim == 1:
                result = result.reshape(-1, 1)
            factor_names = [f"{prefix}_{i}" for i in range(result.shape[1])]
        
        # Validate shape
        if result.shape[0] != n_assets:
            raise ValueError(
                f"Number of assets mismatch: returns has {n_assets} "
                f"but {prefix} has {result.shape[0]}"
            )
        
        return result, factor_names
    
    def _validate_data(
        self,
        returns: np.ndarray,
        risk_factors: np.ndarray,
        alpha_factors: np.ndarray
    ):
        """Validate data dimensions and check for invalid values."""
        n_assets = len(returns)
        
        # Check dimensions
        if risk_factors.shape[0] != n_assets:
            raise ValueError(
                f"Risk factors dimension mismatch: expected {n_assets} assets, "
                f"got {risk_factors.shape[0]}"
            )
        
        if alpha_factors.shape[0] != n_assets:
            raise ValueError(
                f"Alpha factors dimension mismatch: expected {n_assets} assets, "
                f"got {alpha_factors.shape[0]}"
            )
        
        # Check for NaN/Inf
        if not np.all(np.isfinite(returns)):
            raise ValueError("Returns contain NaN or Inf values")
        
        if not np.all(np.isfinite(risk_factors)):
            raise ValueError("Risk factors contain NaN or Inf values")
        
        if not np.all(np.isfinite(alpha_factors)):
            raise ValueError("Alpha factors contain NaN or Inf values")
        
        # Check for sufficient variation
        if np.std(returns) < 1e-10:
            raise ValueError("Returns have zero or near-zero variance")
    
    def standardize_factors(
        self,
        factors: np.ndarray,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Standardize factors to zero mean and unit variance.
        
        Parameters
        ----------
        factors : ndarray
            Factor matrix to standardize
        mean : ndarray, optional
            Pre-computed means (for applying same transformation)
        std : ndarray, optional
            Pre-computed standard deviations
            
        Returns
        -------
        standardized : ndarray
            Standardized factors
        mean : ndarray
            Means used for standardization
        std : ndarray
            Standard deviations used
        """
        if mean is None:
            mean = np.mean(factors, axis=0)
        if std is None:
            std = np.std(factors, axis=0)
            std[std < 1e-10] = 1.0  # Avoid division by zero
        
        standardized = (factors - mean) / std
        return standardized, mean, std
    
    def create_time_series_dataset(
        self,
        returns_panel: pd.DataFrame,
        risk_factors_panel: Dict[str, pd.DataFrame],
        alpha_factors_panel: Dict[str, pd.DataFrame],
        forward_periods: int = 1
    ) -> List[StrategyData]:
        """
        Create a time series of StrategyData objects from panel data.
        
        Parameters
        ----------
        returns_panel : DataFrame
            Panel of returns with DatetimeIndex and asset columns
        risk_factors_panel : dict of DataFrames
            Dictionary of risk factor panels
        alpha_factors_panel : dict of DataFrames
            Dictionary of alpha factor panels
        forward_periods : int
            Number of periods to shift returns forward
            
        Returns
        -------
        list of StrategyData
            Time series of prepared data
        """
        # Ensure all panels have the same index and columns
        dates = returns_panel.index[:-forward_periods]  # Exclude last periods
        asset_ids = returns_panel.columns.tolist()
        
        dataset = []
        
        for t in dates:
            # Get forward returns
            future_idx = returns_panel.index.get_loc(t) + forward_periods
            if future_idx >= len(returns_panel.index):
                break
            
            future_date = returns_panel.index[future_idx]
            returns_t = returns_panel.loc[future_date].values
            
            # Get current risk factors
            risk_factors_t = {}
            for name, df in risk_factors_panel.items():
                risk_factors_t[name] = df.loc[t]
            
            # Get current alpha factors
            alpha_factors_t = {}
            for name, df in alpha_factors_panel.items():
                alpha_factors_t[name] = df.loc[t]
            
            # Prepare data
            data = self.prepare_data(
                returns=returns_t,
                risk_factors=risk_factors_t,
                alpha_factors=alpha_factors_t,
                asset_ids=asset_ids,
                timestamp=t
            )
            
            dataset.append(data)
        
        return dataset
