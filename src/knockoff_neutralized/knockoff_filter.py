"""
Conditional Knockoff Filter Module

Implements Model-X knockoff filters for conditional feature selection.
This allows identification of alpha factors that provide predictive power
beyond what is explained by known risk factors.

Key steps:
1. Model the conditional distribution A|F (alpha factors given risk factors)
2. Generate knockoff variables Ã with proper properties
3. Run feature selection on augmented data [F, A, Ã]
4. Apply FDR control using knockoff+ algorithm
"""

import numpy as np
from scipy import stats
from scipy.linalg import cholesky, solve_triangular
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional
import warnings


class ConditionalKnockoffFilter:
    """
    Implements conditional knockoff filters for feature selection.
    
    The goal is to find which alpha factors A can predict returns Y,
    even after accounting for all information in risk factors F.
    
    This uses the Model-X knockoff framework with conditional independence:
    - Generate Ã such that [F, Ã] ⊥ Y | [F, A]
    - Ã has same covariance structure as A
    - Ã has same relationship with F as A does
    """
    
    def __init__(
        self,
        fdr_target: float = 0.10,
        offset: int = 1,
        random_state: Optional[int] = None
    ):
        """
        Initialize the conditional knockoff filter.
        
        Parameters
        ----------
        fdr_target : float
            Target false discovery rate (e.g., 0.10 for 10%)
        offset : int
            Offset parameter for knockoff+ (1 is more conservative)
        random_state : int, optional
            Random seed for reproducibility
        """
        self.fdr_target = fdr_target
        self.offset = offset
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        # Fitted parameters
        self.selected_indices_ = None
        self.selected_names_ = None
        self.w_statistics_ = None
        self.threshold_ = None
        
        # Conditional distribution parameters
        self.mu_A_ = None
        self.mu_F_ = None
        self.Sigma_AA_ = None
        self.Sigma_AF_ = None
        self.Sigma_FF_ = None
        self.Sigma_AA_F_ = None  # Conditional covariance
        
    def fit(
        self,
        Y: np.ndarray,
        F: np.ndarray,
        A: np.ndarray,
        alpha_factor_names: Optional[List[str]] = None
    ) -> 'ConditionalKnockoffFilter':
        """
        Fit the conditional knockoff filter to select alpha factors.
        
        Parameters
        ----------
        Y : ndarray, shape (n_samples,)
            Target variable (forward returns)
        F : ndarray, shape (n_samples, n_risk_factors)
            Known risk factors to condition on
        A : ndarray, shape (n_samples, n_alpha_factors)
            Candidate alpha factors to select from
        alpha_factor_names : list of str, optional
            Names of alpha factors
            
        Returns
        -------
        self : ConditionalKnockoffFilter
            Fitted filter
        """
        n_samples, n_alpha = A.shape
        n_risk = F.shape[1]
        
        if alpha_factor_names is None:
            alpha_factor_names = [f"alpha_{i}" for i in range(n_alpha)]
        
        # Step 1: Estimate conditional distribution A|F
        self._fit_conditional_distribution(F, A)
        
        # Step 2: Generate conditional knockoffs
        A_tilde = self._generate_conditional_knockoffs(F, A)
        
        # Step 3: Run feature selection on [F, A, A_tilde]
        # Note: We don't penalize F coefficients
        X_augmented = np.hstack([F, A, A_tilde])
        
        # Use LassoCV for automatic lambda selection
        # We'll use a custom penalty that doesn't penalize F
        w_statistics = self._compute_knockoff_statistics(
            Y, F, A, A_tilde
        )
        
        # Step 4: Apply knockoff+ filter
        selected_indices = self._apply_knockoff_filter(w_statistics)
        
        # Store results
        self.w_statistics_ = w_statistics
        self.selected_indices_ = selected_indices
        self.selected_names_ = [alpha_factor_names[i] for i in selected_indices]
        
        return self
    
    def _fit_conditional_distribution(self, F: np.ndarray, A: np.ndarray):
        """
        Estimate parameters of the conditional distribution A|F.
        
        Assumes Gaussian: A|F ~ N(μ_A + Σ_AF Σ_FF^{-1} (F - μ_F), Σ_AA|F)
        where Σ_AA|F = Σ_AA - Σ_AF Σ_FF^{-1} Σ_FA
        """
        # Compute means
        self.mu_F_ = np.mean(F, axis=0)
        self.mu_A_ = np.mean(A, axis=0)
        
        # Compute covariances
        F_centered = F - self.mu_F_
        A_centered = A - self.mu_A_
        
        n_samples = F.shape[0]
        
        self.Sigma_FF_ = (F_centered.T @ F_centered) / n_samples
        self.Sigma_AA_ = (A_centered.T @ A_centered) / n_samples
        self.Sigma_AF_ = (A_centered.T @ F_centered) / n_samples
        self.Sigma_FA_ = self.Sigma_AF_.T
        
        # Add small ridge for numerical stability
        ridge = 1e-6
        self.Sigma_FF_ += ridge * np.eye(self.Sigma_FF_.shape[0])
        self.Sigma_AA_ += ridge * np.eye(self.Sigma_AA_.shape[0])
        
        # Compute conditional covariance: Σ_AA|F = Σ_AA - Σ_AF Σ_FF^{-1} Σ_FA
        Sigma_FF_inv = np.linalg.inv(self.Sigma_FF_)
        self.Sigma_AA_F_ = self.Sigma_AA_ - self.Sigma_AF_ @ Sigma_FF_inv @ self.Sigma_FA_
        
        # Ensure positive definite
        self.Sigma_AA_F_ = self._ensure_positive_definite(self.Sigma_AA_F_)
    
    def _generate_conditional_knockoffs(
        self,
        F: np.ndarray,
        A: np.ndarray
    ) -> np.ndarray:
        """
        Generate conditional knockoff variables Ã.
        
        Uses the equi-correlated construction with SDP optimization
        for the S matrix (simplified version here uses equi-correlated).
        """
        n_samples, n_alpha = A.shape
        
        # Compute S matrix using equi-correlated construction
        S = self._compute_s_matrix(self.Sigma_AA_F_)
        
        # Generate knockoffs: Ã = A - A Σ^{-1} S + noise
        # where noise ~ N(0, S)
        
        Sigma_inv = np.linalg.inv(self.Sigma_AA_F_)
        
        # Compute conditional mean of Ã given A and F
        # Ã|A,F = A - A Σ^{-1} S + noise
        A_centered = A - self.mu_A_
        F_centered = F - self.mu_F_
        
        # Adjust for conditioning on F
        # Conditional mean: μ_A + Σ_AF Σ_FF^{-1} (F - μ_F)
        Sigma_FF_inv = np.linalg.inv(self.Sigma_FF_)
        # Shape: (n_samples, n_risk) @ (n_risk, n_risk) @ (n_risk, n_alpha) = (n_samples, n_alpha)
        A_mean_given_F = self.mu_A_ + (F_centered @ Sigma_FF_inv @ self.Sigma_FA_)
        
        # Center A appropriately
        A_centered_given_F = A - A_mean_given_F
        
        # Knockoff construction
        mean_tilde = A - A_centered_given_F @ Sigma_inv @ S
        
        # Generate noise with covariance 2S - S Σ^{-1} S
        try:
            noise_cov = 2 * S - S @ Sigma_inv @ S
            noise_cov = self._ensure_positive_definite(noise_cov)
            L = cholesky(noise_cov, lower=True)
            noise = self.rng.randn(n_samples, n_alpha) @ L.T
        except np.linalg.LinAlgError:
            # Fallback: use simpler construction
            warnings.warn("Cholesky failed, using simpler knockoff construction")
            L = cholesky(S, lower=True)
            noise = self.rng.randn(n_samples, n_alpha) @ L.T
        
        A_tilde = mean_tilde + noise
        
        return A_tilde
    
    def _compute_s_matrix(self, Sigma: np.ndarray) -> np.ndarray:
        """
        Compute the S matrix for knockoff construction.
        
        Uses equi-correlated construction: S = (1 - γ) * Σ
        where γ is chosen to ensure 2S - S Σ^{-1} S is PSD.
        """
        n = Sigma.shape[0]
        
        # Find eigenvalues
        eigvals = np.linalg.eigvalsh(Sigma)
        lambda_min = np.min(eigvals)
        
        # Set γ to ensure positive definiteness
        # A conservative choice is γ = 1 / (2 * max_eigval)
        gamma = 1.0 / (2.0 * np.max(eigvals))
        gamma = min(gamma, 1.0 - 1e-3)  # Ensure < 1
        
        # Equi-correlated construction
        s_diag = np.minimum((1.0 - gamma) * np.diag(Sigma), lambda_min)
        s_diag = np.maximum(s_diag, 1e-6)  # Ensure positive
        
        S = np.diag(s_diag)
        
        return S
    
    def _ensure_positive_definite(
        self,
        matrix: np.ndarray,
        epsilon: float = 1e-6
    ) -> np.ndarray:
        """Ensure matrix is positive definite by adding ridge if needed."""
        eigvals = np.linalg.eigvalsh(matrix)
        if np.min(eigvals) < epsilon:
            ridge = epsilon - np.min(eigvals) + epsilon
            matrix = matrix + ridge * np.eye(matrix.shape[0])
        return matrix
    
    def _compute_knockoff_statistics(
        self,
        Y: np.ndarray,
        F: np.ndarray,
        A: np.ndarray,
        A_tilde: np.ndarray
    ) -> np.ndarray:
        """
        Compute knockoff statistics W_j = |β_j| - |β̃_j|.
        
        We run Lasso on [F, A, Ã] where F is not penalized.
        """
        n_samples, n_alpha = A.shape
        n_risk = F.shape[1]
        
        # Standardize for Lasso
        scaler_F = StandardScaler()
        scaler_A = StandardScaler()
        
        F_scaled = scaler_F.fit_transform(F)
        A_scaled = scaler_A.fit_transform(A)
        A_tilde_scaled = scaler_A.transform(A_tilde)  # Use same scaling as A
        
        # We need to handle F differently (no penalty)
        # Strategy: regress Y on F first, then use residuals
        
        # Step 1: Regress Y on F (OLS, no penalty)
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(F_scaled, Y)
        Y_resid = Y - lr.predict(F_scaled)
        
        # Step 2: Run Lasso on residuals with [A, Ã]
        X_alpha_augmented = np.hstack([A_scaled, A_tilde_scaled])
        
        # Use LassoCV for automatic lambda selection
        lasso = LassoCV(cv=5, random_state=self.random_state, max_iter=10000)
        lasso.fit(X_alpha_augmented, Y_resid)
        
        coefs = lasso.coef_
        
        # Split coefficients
        beta_A = coefs[:n_alpha]
        beta_A_tilde = coefs[n_alpha:]
        
        # Compute knockoff statistics
        w_statistics = np.abs(beta_A) - np.abs(beta_A_tilde)
        
        return w_statistics
    
    def _apply_knockoff_filter(self, w_statistics: np.ndarray) -> np.ndarray:
        """
        Apply the knockoff+ filter to control FDR.
        
        Parameters
        ----------
        w_statistics : ndarray
            Knockoff statistics W_j for each feature
            
        Returns
        -------
        selected : ndarray
            Indices of selected features
        """
        n_features = len(w_statistics)
        
        if n_features == 0:
            return np.array([], dtype=int)
        
        # Sort W statistics in descending order
        sorted_indices = np.argsort(-w_statistics)  # Descending
        sorted_w = w_statistics[sorted_indices]
        
        # Apply knockoff+ threshold
        # For each threshold t, compute FDR estimate:
        # FDR(t) = (1 + #{j: W_j <= -t}) / max(1, #{j: W_j >= t})
        
        selected = []
        
        for i in range(n_features):
            t = sorted_w[i]
            
            if t <= 0:
                break  # All remaining are negative
            
            # Count positives and negatives
            n_positive = np.sum(w_statistics >= t)
            n_negative = np.sum(w_statistics <= -t)
            
            # Knockoff+ FDR estimate
            fdr_estimate = (self.offset + n_negative) / max(1, n_positive)
            
            if fdr_estimate <= self.fdr_target:
                # Select all features with W >= t
                selected = sorted_indices[w_statistics[sorted_indices] >= t]
                self.threshold_ = t
                break
        
        if len(selected) == 0:
            selected = np.array([], dtype=int)
        
        return selected
    
    def get_selected_features(self) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Get selected features after fitting.
        
        Returns
        -------
        indices : ndarray
            Indices of selected features
        names : list of str
            Names of selected features
        w_statistics : ndarray
            Knockoff statistics for all features
        """
        if self.selected_indices_ is None:
            raise RuntimeError("Must call fit() before getting selected features")
        
        return self.selected_indices_, self.selected_names_, self.w_statistics_
    
    def transform(self, A: np.ndarray) -> np.ndarray:
        """
        Transform alpha factors by selecting only the chosen features.
        
        Parameters
        ----------
        A : ndarray, shape (n_samples, n_alpha_factors)
            Alpha factors
            
        Returns
        -------
        A_selected : ndarray, shape (n_samples, n_selected)
            Selected alpha factors
        """
        if self.selected_indices_ is None:
            raise RuntimeError("Must call fit() before transform()")
        
        if len(self.selected_indices_) == 0:
            return np.zeros((A.shape[0], 0))
        
        return A[:, self.selected_indices_]
    
    def fit_transform(
        self,
        Y: np.ndarray,
        F: np.ndarray,
        A: np.ndarray,
        alpha_factor_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """Fit the filter and return selected features."""
        self.fit(Y, F, A, alpha_factor_names)
        return self.transform(A)
