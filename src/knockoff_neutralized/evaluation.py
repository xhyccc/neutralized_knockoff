"""
Comprehensive evaluation engine referenced in evaluation_plan.md.

This module defines a structured backtesting harness capable of running the
"horse race" of strategies (S1-S8) described in the plan.  The design focuses on
flexibility: the engine operates on point-in-time cross-sectional snapshots and
does not assume a specific data source.  Users can plug in true market data or
fall back to synthetic panels for rapid experimentation.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression

from .data_preparation import StrategyData
from .knockoff_filter import ConditionalKnockoffFilter
from .portfolio_optimizer import PortfolioOptimizer


TRANSACTION_COST = 0.001  # 10 bps per plan
PORTFOLIO_SMOOTHING = 0.35
LONG_SHORT_FRACTION = 0.2
DAYS_PER_YEAR = 252


@dataclass
class EvaluationConfig:
    """Configuration for the rolling evaluation."""

    training_window_days: int = 252  # roughly one trading year
    rebalance_frequency_days: int = 1  # daily by default
    cost_per_dollar: float = TRANSACTION_COST
    smoothing: float = PORTFOLIO_SMOOTHING
    long_short_fraction: float = LONG_SHORT_FRACTION
    target_beta: Optional[float] = None  # Target beta for long-only portfolios (None = use beta_neutral=True)
    beta_tolerance: float = 0.15  # Tolerance for target_beta constraint


@dataclass
class StrategyMetrics:
    """Performance diagnostics for a strategy."""

    name: str
    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe: float
    max_drawdown: float
    calmar: float
    avg_turnover: float
    realized_factor_exposure: Dict[str, float]
    returns: pd.Series
    weights: pd.DataFrame


class StrategyBase:
    """Interface implemented by all evaluation strategies."""

    def __init__(self, name: str):
        self.name = name
        self._asset_ids: Optional[List[str]] = None
        self._risk_factor_names: Optional[List[str]] = None
        self._alpha_factor_names: Optional[List[str]] = None

    def initialise(self, template: StrategyData) -> None:
        self._asset_ids = template.asset_ids
        self._risk_factor_names = template.risk_factor_names
        self._alpha_factor_names = template.alpha_factor_names
        self._do_initialise(template)

    # pylint: disable=unused-argument
    def _do_initialise(self, template: StrategyData) -> None:
        """Hook for subclasses."""

    def fit(self, training_data: Sequence[StrategyData]) -> None:
        """Fit model parameters using the rolling history."""

    def generate_weights(
        self,
        snapshot: StrategyData,
        previous_weights: np.ndarray,
        config: EvaluationConfig,
    ) -> np.ndarray:
        raise NotImplementedError

    # Utilities -----------------------------------------------------------------
    @property
    def asset_ids(self) -> List[str]:
        if self._asset_ids is None:
            raise RuntimeError("Strategy not initialised")
        return self._asset_ids

    @property
    def risk_factor_names(self) -> List[str]:
        if self._risk_factor_names is None:
            raise RuntimeError("Strategy not initialised")
        return self._risk_factor_names

    @property
    def alpha_factor_names(self) -> List[str]:
        if self._alpha_factor_names is None:
            raise RuntimeError("Strategy not initialised")
        return self._alpha_factor_names


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _stack_training_matrices(training_data: Sequence[StrategyData]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack training data into 2D matrices."""

    y_list = []
    f_list = []
    a_list = []

    for snap in training_data:
        y_list.append(snap.returns.reshape(-1, 1))
        f_list.append(snap.risk_factors)
        a_list.append(snap.alpha_factors)

    y_train = np.vstack(y_list).flatten()
    f_train = np.vstack(f_list)
    a_train = np.vstack(a_list)
    return y_train, f_train, a_train


def _standardise(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-8] = 1.0
    standardized = (matrix - mean) / std
    return standardized, mean, std


def _rank_to_signal(values: np.ndarray, long_short_fraction: float) -> np.ndarray:
    """Convert raw scores into long/short weights."""

    values = np.nan_to_num(values, nan=0.0)
    n_assets = len(values)
    n_long = max(1, int(np.floor(n_assets * long_short_fraction)))
    n_short = n_long

    ranking = np.argsort(values)
    weights = np.zeros(n_assets)

    short_indices = ranking[:n_short]
    long_indices = ranking[-n_long:]

    weights[long_indices] = 1.0 / n_long
    weights[short_indices] = -1.0 / n_short
    return weights


def _apply_smoothing(current: np.ndarray, target: np.ndarray, smoothing: float) -> np.ndarray:
    return (1 - smoothing) * current + smoothing * target


def _scores_to_long_only(scores: np.ndarray, top_fraction: float) -> np.ndarray:
    """Map alpha scores to long-only weights that sum to one."""

    scores = np.asarray(scores, dtype=float)
    n_assets = scores.size
    if n_assets == 0:
        return scores

    finite_mask = np.isfinite(scores)
    if not np.any(finite_mask):
        return np.ones(n_assets) / n_assets

    sanitized = scores.copy()
    min_finite = np.min(scores[finite_mask])
    sanitized[~finite_mask] = min_finite - 1.0

    fraction = float(np.clip(top_fraction, 0.0, 1.0))
    n_select = max(1, int(np.floor(n_assets * fraction))) if fraction > 0 else max(1, n_assets)
    if n_select >= n_assets:
        top_indices = np.arange(n_assets)
    else:
        cutoff_index = np.argpartition(-sanitized, n_select - 1)[:n_select]
        # Sort for stability
        top_indices = cutoff_index[np.argsort(-sanitized[cutoff_index])]

    weights = np.zeros(n_assets)
    selected_scores = sanitized[top_indices]
    shifted = selected_scores - selected_scores.min()
    shifted = np.maximum(shifted, 0.0)

    if shifted.sum() <= 1e-12:
        weights[top_indices] = 1.0 / len(top_indices)
    else:
        weights[top_indices] = shifted / shifted.sum()

    total = weights.sum()
    if total <= 1e-12:
        return np.ones(n_assets) / n_assets
    return weights / total


def _ensure_long_only(weights: np.ndarray) -> np.ndarray:
    """Project weights to the long-only simplex."""

    projected = np.maximum(np.asarray(weights, dtype=float), 0.0)
    total = projected.sum()
    if total <= 1e-12:
        n_assets = projected.size
        if n_assets == 0:
            return projected
        return np.ones(n_assets) / n_assets
    return projected / total


def _project_out_factors(scores: np.ndarray, factor_matrix: np.ndarray) -> np.ndarray:
    """Remove linear exposure to risk factors from alpha scores."""

    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores

    if factor_matrix.size == 0:
        return scores - np.mean(scores)

    X = np.asarray(factor_matrix, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    X = np.nan_to_num(X)
    y = np.nan_to_num(scores)

    X_centered = X - X.mean(axis=0, keepdims=True)
    y_centered = y - y.mean()

    try:
        coefs, *_ = np.linalg.lstsq(X_centered, y_centered, rcond=None)
        residual = y_centered - X_centered @ coefs
    except np.linalg.LinAlgError:  # pragma: no cover
        residual = y_centered

    return residual


# ---------------------------------------------------------------------------
# Strategy Implementations S1-S8
# ---------------------------------------------------------------------------


class MarketIndexStrategy(StrategyBase):
    """S1: buy-and-hold proxy using cross-sectional mean return."""

    def __init__(self):
        super().__init__("S1_MarketIndex")
        self._weights: Optional[np.ndarray] = None

    def _do_initialise(self, template: StrategyData) -> None:
        n = len(template.asset_ids)
        self._weights = np.ones(n) / n

    def fit(self, training_data: Sequence[StrategyData]) -> None:  # noqa: D401
        # No fitting required
        return

    def generate_weights(self, snapshot: StrategyData, previous_weights: np.ndarray, config: EvaluationConfig) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("Strategy not initialised")
        return self._weights


class EqualWeightLongOnlyStrategy(StrategyBase):
    """S2: equal-weight long-only portfolio."""

    def __init__(self):
        super().__init__("S2_EqualWeight")

    def generate_weights(self, snapshot: StrategyData, previous_weights: np.ndarray, config: EvaluationConfig) -> np.ndarray:
        n = len(snapshot.asset_ids)
        return np.ones(n) / n


class _KnockoffBase(StrategyBase):
    def __init__(
        self,
        name: str,
        condition_on_factors: bool,
        fdr_target: float = 0.1,
        random_state: int = 42,
        fallback_top_k: int = 10,
    ):
        super().__init__(name)
        self.condition_on_factors = condition_on_factors
        self.selected_indices: List[int] = []
        self.filter_: Optional[ConditionalKnockoffFilter] = None
        self.fdr_target = fdr_target
        self.random_state = random_state
        self.fallback_top_k = fallback_top_k
        self.alpha_coef_: Optional[np.ndarray] = None
        self.alpha_mean_: Optional[np.ndarray] = None
        self.alpha_std_: Optional[np.ndarray] = None

    def fit(self, training_data: Sequence[StrategyData]) -> None:
        y_train, f_train, a_train = _stack_training_matrices(training_data)

        if not self.condition_on_factors:
            f_train = np.zeros((f_train.shape[0], 1))

        self.filter_ = ConditionalKnockoffFilter(fdr_target=self.fdr_target, random_state=self.random_state)
        self.filter_.fit(y_train, f_train, a_train, self.alpha_factor_names)
        raw_selected = getattr(self.filter_, "selected_indices_", None)
        if raw_selected is None:
            self.selected_indices = []
        else:
            self.selected_indices = list(np.atleast_1d(raw_selected).astype(int))

        if not self.selected_indices:
            residuals = y_train.copy()
            if self.condition_on_factors and f_train.size:
                reg = LinearRegression()
                reg.fit(f_train, y_train)
                residuals = y_train - reg.predict(f_train)

            residuals -= residuals.mean()
            alpha_centered = a_train - a_train.mean(axis=0)
            numerator = residuals @ alpha_centered
            denom = (np.sqrt((residuals ** 2).sum()) * np.sqrt((alpha_centered ** 2).sum(axis=0)))
            with np.errstate(invalid="ignore", divide="ignore"):
                correlations = numerator / denom
            correlations = np.nan_to_num(correlations)
            top_k = max(1, min(self.fallback_top_k, correlations.size))
            self.selected_indices = list(np.argsort(-np.abs(correlations))[:top_k])

        if self.selected_indices:
            alpha_subset = a_train[:, self.selected_indices]
            alpha_subset_std, mean, std = _standardise(alpha_subset)
            self.alpha_mean_ = mean
            self.alpha_std_ = std

            residuals = y_train.copy()
            if self.condition_on_factors and f_train.size:
                reg_factors = LinearRegression()
                reg_factors.fit(f_train, y_train)
                residuals = y_train - reg_factors.predict(f_train)

            lasso = LassoCV(cv=5, random_state=self.random_state, max_iter=20000)
            lasso.fit(alpha_subset_std, residuals)
            coef = lasso.coef_

            self.alpha_coef_ = coef
        else:
            self.alpha_coef_ = None
            self.alpha_mean_ = None
            self.alpha_std_ = None

    def _build_alpha_scores(self, snapshot: StrategyData) -> np.ndarray:
        if not self.selected_indices:
            return np.zeros(len(snapshot.asset_ids))

        selected_alpha = snapshot.alpha_factors[:, self.selected_indices]
        if self.alpha_mean_ is not None and self.alpha_std_ is not None:
            std = np.where(np.abs(self.alpha_std_) < 1e-6, 1.0, self.alpha_std_)
            selected_alpha = (selected_alpha - self.alpha_mean_) / std
        else:
            selected_alpha, _, _ = _standardise(selected_alpha)

        if self.alpha_coef_ is not None:
            scores = selected_alpha @ self.alpha_coef_
        else:
            scores = np.nanmean(selected_alpha, axis=1)
        return scores


class VanillaKnockoffSimpleLongShortStrategy(_KnockoffBase):
    def __init__(self, fdr_target: float = 0.1, random_state: int = 42, fallback_top_k: int = 10):
        super().__init__(
            "S3_VK_SimpleLS",
            condition_on_factors=False,
            fdr_target=fdr_target,
            random_state=random_state,
            fallback_top_k=fallback_top_k,
        )

    def generate_weights(self, snapshot: StrategyData, previous_weights: np.ndarray, config: EvaluationConfig) -> np.ndarray:
        scores = self._build_alpha_scores(snapshot)
        weights = _scores_to_long_only(scores, config.long_short_fraction)
        return _ensure_long_only(weights)


class VanillaKnockoffFactorNeutralStrategy(_KnockoffBase):
    def __init__(
        self,
        fdr_target: float = 0.1,
        random_state: int = 42,
        fallback_top_k: int = 10,
        risk_aversion: float = 1.0,
    ):
        super().__init__(
            "S4_VK_FactorNeutral",
            condition_on_factors=False,
            fdr_target=fdr_target,
            random_state=random_state,
            fallback_top_k=fallback_top_k,
        )
        self.optimizer = PortfolioOptimizer(
            risk_aversion=risk_aversion,
            max_leverage=1.0,
            neutralize_factors=True,
            transaction_cost=TRANSACTION_COST,
            long_only=True,
            beta_neutral=False,  # Will use target_beta from config instead
        )

    def generate_weights(self, snapshot: StrategyData, previous_weights: np.ndarray, config: EvaluationConfig) -> np.ndarray:
        scores = self._build_alpha_scores(snapshot)
        scores = _project_out_factors(scores, snapshot.risk_factors)
        if not np.any(np.abs(scores) > 1e-12):
            return _ensure_long_only(previous_weights)

        # Apply target_beta from config
        self.optimizer.target_beta = config.target_beta
        self.optimizer.beta_tolerance = config.beta_tolerance

        factor_matrix = snapshot.risk_factors
        result = self.optimizer.optimize(scores, factor_matrix, current_weights=previous_weights)
        target = result.weights if result.success else previous_weights
        smoothed = _apply_smoothing(previous_weights, target, config.smoothing)
        return _ensure_long_only(smoothed)


class ConditionalKnockoffSimpleLongShortStrategy(_KnockoffBase):
    def __init__(
        self,
        fdr_target: float = 0.15,
        random_state: int = 84,
        fallback_top_k: int = 10,
        long_short_fraction: float = 0.12,
    ):
        super().__init__(
            "S5_CK_SimpleLS",
            condition_on_factors=True,
            fdr_target=fdr_target,
            random_state=random_state,
            fallback_top_k=fallback_top_k,
        )
        self.custom_long_short_fraction = long_short_fraction

    def generate_weights(self, snapshot: StrategyData, previous_weights: np.ndarray, config: EvaluationConfig) -> np.ndarray:
        scores = self._build_alpha_scores(snapshot)
        ls_fraction = self.custom_long_short_fraction or config.long_short_fraction
        weights = _scores_to_long_only(scores, ls_fraction)
        return _ensure_long_only(weights)


class ConditionalKnockoffFactorNeutralStrategy(_KnockoffBase):
    def __init__(
        self,
        fdr_target: float = 0.12,
        random_state: int = 99,
        fallback_top_k: int = 10,
        risk_aversion: float = 0.85,
        smoothing_override: Optional[float] = 0.4,
    ):
        super().__init__(
            "S6_CK_FactorNeutral",
            condition_on_factors=True,
            fdr_target=fdr_target,
            random_state=random_state,
            fallback_top_k=fallback_top_k,
        )
        self.optimizer = PortfolioOptimizer(
            risk_aversion=risk_aversion,
            max_leverage=1.2,
            neutralize_factors=True,
            transaction_cost=TRANSACTION_COST,
            long_only=True,
            beta_neutral=False,  # Will use target_beta from config instead
        )
        self.smoothing_override = smoothing_override

    def generate_weights(self, snapshot: StrategyData, previous_weights: np.ndarray, config: EvaluationConfig) -> np.ndarray:
        scores = self._build_alpha_scores(snapshot)
        if not np.any(np.abs(scores) > 1e-12):
            return _ensure_long_only(previous_weights)

        # Apply target_beta from config
        self.optimizer.target_beta = config.target_beta
        self.optimizer.beta_tolerance = config.beta_tolerance

        scores = _project_out_factors(scores, snapshot.risk_factors)
        factor_matrix = snapshot.risk_factors
        result = self.optimizer.optimize(scores, factor_matrix, current_weights=previous_weights)
        target = result.weights if result.success else previous_weights
        smoothing = self.smoothing_override if self.smoothing_override is not None else config.smoothing
        smoothed = _apply_smoothing(previous_weights, target, smoothing)
        return _ensure_long_only(smoothed)


class LassoFactorNeutralStrategy(StrategyBase):
    def __init__(self):
        super().__init__("S7_Lasso_FactorNeutral")
        self.selected_indices: List[int] = []
        self.optimizer = PortfolioOptimizer(
            risk_aversion=1.0,
            max_leverage=1.0,
            neutralize_factors=True,
            transaction_cost=TRANSACTION_COST,
            long_only=True,
            beta_neutral=False,  # Will use target_beta from config instead
        )

    def fit(self, training_data: Sequence[StrategyData]) -> None:
        y_train, f_train, a_train = _stack_training_matrices(training_data)
        design = np.hstack([f_train, a_train])
        design_std, _, _ = _standardise(design)
        model = LassoCV(cv=5, random_state=42, max_iter=10000)
        model.fit(design_std, y_train)
        coef = model.coef_[f_train.shape[1]:]
        self.selected_indices = list(np.where(np.abs(coef) > 1e-6)[0])

    def generate_weights(self, snapshot: StrategyData, previous_weights: np.ndarray, config: EvaluationConfig) -> np.ndarray:
        if not self.selected_indices:
            return _ensure_long_only(previous_weights)

        # Apply target_beta from config
        self.optimizer.target_beta = config.target_beta
        self.optimizer.beta_tolerance = config.beta_tolerance

        selected_alpha = snapshot.alpha_factors[:, self.selected_indices]
        selected_alpha, _, _ = _standardise(selected_alpha)
        scores = np.nanmean(selected_alpha, axis=1)
        scores = _project_out_factors(scores, snapshot.risk_factors)
        scores_std, _, _ = _standardise(scores.reshape(-1, 1))
        result = self.optimizer.optimize(scores_std.flatten(), snapshot.risk_factors, current_weights=previous_weights)
        target = result.weights if result.success else previous_weights
        smoothed = _apply_smoothing(previous_weights, target, config.smoothing)
        return _ensure_long_only(smoothed)


class CompositeFactorNeutralStrategy(StrategyBase):
    def __init__(self):
        super().__init__("S8_Composite_FactorNeutral")
        self.optimizer = PortfolioOptimizer(
            risk_aversion=1.0,
            max_leverage=1.0,
            neutralize_factors=True,
            transaction_cost=TRANSACTION_COST,
            long_only=True,
            beta_neutral=False,  # Will use target_beta from config instead
        )

    def generate_weights(self, snapshot: StrategyData, previous_weights: np.ndarray, config: EvaluationConfig) -> np.ndarray:
        # Apply target_beta from config
        self.optimizer.target_beta = config.target_beta
        self.optimizer.beta_tolerance = config.beta_tolerance

        alpha_scores, _, _ = _standardise(snapshot.alpha_factors)
        scores = np.nanmean(alpha_scores, axis=1)
        scores = _project_out_factors(scores, snapshot.risk_factors)
        scores_std, _, _ = _standardise(scores.reshape(-1, 1))
        result = self.optimizer.optimize(scores_std.flatten(), snapshot.risk_factors, current_weights=previous_weights)
        target = result.weights if result.success else previous_weights
        smoothed = _apply_smoothing(previous_weights, target, config.smoothing)
        return _ensure_long_only(smoothed)


# ---------------------------------------------------------------------------
# Evaluation Engine
# ---------------------------------------------------------------------------


@dataclass
class StrategyTrack:
    weights: List[np.ndarray] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)
    factor_exposures: List[np.ndarray] = field(default_factory=list)
    turnover: List[float] = field(default_factory=list)
    timestamps: List[pd.Timestamp] = field(default_factory=list)


class EvaluationEngine:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def run(
        self,
        snapshots: Sequence[StrategyData],
        strategies: Iterable[StrategyBase],
    ) -> Dict[str, StrategyMetrics]:
        strategies = list(strategies)
        if not strategies:
            raise ValueError("No strategies supplied")

        for strategy in strategies:
            strategy.initialise(snapshots[0])

        tracks: Dict[str, StrategyTrack] = {s.name: StrategyTrack() for s in strategies}

        previous_weights: Dict[str, np.ndarray] = {
            s.name: np.zeros(len(snapshots[0].asset_ids)) for s in strategies
        }

        window = self.config.training_window_days
        rebalance = max(1, self.config.rebalance_frequency_days)

        for idx in range(window, len(snapshots)):
            train_slice = snapshots[idx - window : idx]
            snapshot = snapshots[idx]
            should_rebalance = (idx - window) % rebalance == 0

            for strategy in strategies:
                current_weights = previous_weights[strategy.name]
                target_weights = current_weights
                turnover = 0.0

                if should_rebalance:
                    strategy.fit(train_slice)
                    target_weights = strategy.generate_weights(snapshot, current_weights, self.config)
                    turnover = np.sum(np.abs(target_weights - current_weights))
                    previous_weights[strategy.name] = target_weights

                net_return = float(np.dot(target_weights, snapshot.returns) - turnover * self.config.cost_per_dollar)
                exposure = target_weights @ snapshot.risk_factors

                track = tracks[strategy.name]
                track.weights.append(target_weights.copy())
                track.turnover.append(turnover)
                track.returns.append(net_return)
                track.factor_exposures.append(exposure)
                track.timestamps.append(snapshot.timestamp or pd.Timestamp.now())

        results = {
            name: self._build_metrics(name, track, snapshots[0].asset_ids, snapshots[0].risk_factor_names)
            for name, track in tracks.items()
        }
        return results

    @staticmethod
    def _build_metrics(
        name: str,
        track: StrategyTrack,
        asset_ids: List[str],
        risk_factor_names: List[str],
    ) -> StrategyMetrics:
        if not track.returns:
            empty_index = pd.Index([], name="timestamp")
            empty_series = pd.Series(data=[], index=pd.Index([], name="timestamp"), dtype=float, name=name)
            empty_weights = pd.DataFrame(columns=asset_ids)
            return StrategyMetrics(
                name=name,
                total_return=0.0,
                annualized_return=0.0,
                annualized_vol=0.0,
                sharpe=0.0,
                max_drawdown=0.0,
                calmar=np.inf,
                avg_turnover=0.0,
                realized_factor_exposure={factor: 0.0 for factor in risk_factor_names},
                returns=empty_series,
                weights=empty_weights,
            )

        returns = pd.Series(track.returns, index=track.timestamps, name=name)
        weights = pd.DataFrame(track.weights, index=track.timestamps, columns=asset_ids)
        exposures = np.vstack(track.factor_exposures) if track.factor_exposures else np.zeros((0, len(risk_factor_names)))
        turnover = np.array(track.turnover)

        cumulative = (1 + returns).cumprod()
        total_return = float(cumulative.iloc[-1] - 1)
        n_periods = len(returns)
        annualized_return = (1 + total_return) ** (DAYS_PER_YEAR / n_periods) - 1 if n_periods > 0 else 0.0
        annualized_vol = float(returns.std(ddof=1) * np.sqrt(DAYS_PER_YEAR)) if n_periods > 1 else 0.0
        sharpe = annualized_return / annualized_vol if annualized_vol > 1e-8 else 0.0
        running_max = cumulative.cummax()
        max_drawdown = float(((cumulative - running_max) / running_max).min())
        calmar = annualized_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
        avg_turnover = float(turnover.mean()) if turnover.size > 0 else 0.0

        realized_factor_exposure = {
            name: float(np.mean(exposures[:, idx])) if exposures.size else 0.0
            for idx, name in enumerate(risk_factor_names)
        }

        return StrategyMetrics(
            name=name,
            total_return=total_return,
            annualized_return=annualized_return,
            annualized_vol=annualized_vol,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            calmar=calmar,
            avg_turnover=avg_turnover,
            realized_factor_exposure=realized_factor_exposure,
            returns=returns,
            weights=weights,
        )


# Convenience -----------------------------------------------------------------


def default_strategies(
    conditional_knockoff_params: Optional[Dict[str, float]] = None,
) -> List[StrategyBase]:
    conditional_knockoff_params = conditional_knockoff_params or {}
    return [
        MarketIndexStrategy(),
        EqualWeightLongOnlyStrategy(),
        VanillaKnockoffSimpleLongShortStrategy(),
        VanillaKnockoffFactorNeutralStrategy(),
        ConditionalKnockoffSimpleLongShortStrategy(),
        ConditionalKnockoffFactorNeutralStrategy(**conditional_knockoff_params),
        LassoFactorNeutralStrategy(),
        CompositeFactorNeutralStrategy(),
    ]


__all__ = [
    "EvaluationConfig",
    "StrategyMetrics",
    "StrategyBase",
    "EvaluationEngine",
    "default_strategies",
]
