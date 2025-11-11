"""Comprehensive evaluation driver for the S1-S8 horse race.

This script instantiates the evaluation engine defined in
`src/knockoff_neutralized/evaluation.py` and runs it against monthly panels built
from Yahoo Finance data.  The dataset mirrors the structure expected by the
research plan (returns, risk factors, alpha factors) so the same pipeline can be
pointed at richer fundamentals once they are available.
"""
from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from knockoff_neutralized.data_preparation import DataPreparation
from knockoff_neutralized.evaluation import (
    ConditionalKnockoffFactorNeutralStrategy,
    EvaluationConfig,
    EvaluationEngine,
    StrategyData,
    StrategyMetrics,
    default_strategies,
)
from knockoff_neutralized.yfinance_loader import DEFAULT_TICKERS, load_yfinance_panels


RESULTS_DIR = Path("evaluation_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GRID_LIBRARY: Dict[str, Dict[str, Sequence[float]]] = {
    "small": {
        "fdr_target": [0.05, 0.10],
        "risk_aversion": [0.6, 0.85],
        "smoothing_override": [0.25, 0.35],
        "fallback_top_k": [8],
    },
    "medium": {
        "fdr_target": [0.06, 0.08, 0.1, 0.12],
        "risk_aversion": [0.5, 0.65, 0.85],
        "smoothing_override": [0.2, 0.3, 0.4],
        "fallback_top_k": [6, 8],
    },
    "large": {
        "fdr_target": [0.05, 0.06, 0.08, 0.1, 0.12, 0.15],
        "risk_aversion": [0.45, 0.55, 0.7, 0.85, 1.0],
        "smoothing_override": [0.15, 0.25, 0.35, 0.45],
        "fallback_top_k": [6, 8, 10],
    },
    "xlarge": {
        "fdr_target": [0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2],
        "risk_aversion": [0.35, 0.45, 0.55, 0.7, 0.85, 1.05],
        "smoothing_override": [0.1, 0.2, 0.3, 0.4, 0.5],
        "fallback_top_k": [5, 6, 8, 10],
    },
}


# ---------------------------------------------------------------------------
# Evaluation workflow --------------------------------------------------------
# ---------------------------------------------------------------------------


def run_evaluation(
    training_window_days: int = 60,
    rebalance_frequency_days: int = 3,
    cost_per_dollar: float = 0.01,
    smoothing: float = 0.35,
    param_grid: Dict[str, Sequence[Any]] | None = None,
    search_mode: str = "grid",
    search_samples: int | None = None,
    tickers: Optional[Sequence[str]] = None,
    start_date: str = "2014-01-01",
    end_date: str = "2024-12-31",
    target_beta: Optional[float] = None,
    beta_tolerance: float = 0.15,
) -> Dict[str, StrategyMetrics]:
    panels = load_yfinance_panels(tickers=tickers, start=start_date, end=end_date)
    returns_panel = panels.returns_panel
    risk_panel = panels.risk_factors_panel
    alpha_panel = panels.alpha_factors_panel

    dataprep = DataPreparation()
    dataset = dataprep.create_time_series_dataset(
        returns_panel=returns_panel,
        risk_factors_panel=risk_panel,
        alpha_factors_panel=alpha_panel,
        forward_periods=rebalance_frequency_days,
    )

    config = EvaluationConfig(
        training_window_days=training_window_days,
        rebalance_frequency_days=rebalance_frequency_days,
        cost_per_dollar=cost_per_dollar,
        smoothing=smoothing,
        target_beta=target_beta,
        beta_tolerance=beta_tolerance,
    )

    tuned_params, best_stats = tune_conditional_knockoff(
        dataset,
        config,
        param_grid,
        search_mode,
        search_samples,
    )
    if tuned_params:
        sharpe = best_stats.get("sharpe", float("nan"))
        total_return = best_stats.get("total_return", float("nan"))
        print(
            "Best conditional knockoff params:",
            tuned_params,
            f"Sharpe={sharpe:.4f}",
            f"TotalReturn={total_return:.4f}",
        )

    engine = EvaluationEngine(config)
    strategies = default_strategies(conditional_knockoff_params=tuned_params)
    metrics = engine.run(dataset, strategies)
    return metrics


def tune_conditional_knockoff(
    dataset: Sequence[StrategyData],
    config: EvaluationConfig,
    param_grid: Dict[str, Sequence[Any]] | None,
    search_mode: str,
    search_samples: int | None,
) -> tuple[Dict[str, Any], Dict[str, float]]:
    """Grid search for ConditionalKnockoffFactorNeutralStrategy hyperparameters."""

    if not param_grid:
        param_grid = GRID_LIBRARY["medium"]

    combos = list(product(*param_grid.values()))
    keys = list(param_grid.keys())

    if search_mode not in {"grid", "random"}:
        raise ValueError(f"Unknown search_mode '{search_mode}'")

    original_count = len(combos)

    if search_mode == "random" and search_samples:
        sample_size = min(search_samples, original_count)
        rng = np.random.default_rng(123)
        sampled_indices = rng.choice(original_count, size=sample_size, replace=False)
        combos = [combos[idx] for idx in sampled_indices]
        print(f"Random search: sampling {sample_size} out of {original_count} combinations")
    else:
        print(f"Grid search: evaluating {original_count} combinations")

    best_params: Dict[str, Any] = {}
    best_sharpe = -np.inf
    best_total_return = -np.inf

    for combo in combos:
        params = dict(zip(keys, combo))
        strategy = ConditionalKnockoffFactorNeutralStrategy(**params)
        engine = EvaluationEngine(config)
        metrics = engine.run(dataset, [strategy])
        single_metric = next(iter(metrics.values()))
        total_return = single_metric.total_return
        sharpe = single_metric.sharpe

        is_better = False
        if sharpe > best_sharpe + 1e-9:
            is_better = True
        elif abs(sharpe - best_sharpe) <= 1e-9 and total_return > best_total_return:
            is_better = True

        if is_better:
            best_sharpe = sharpe
            best_total_return = total_return
            best_params = params

    return best_params, {"sharpe": best_sharpe, "total_return": best_total_return}


def summarise_metrics(metrics: Dict[str, StrategyMetrics]) -> pd.DataFrame:
    records = []
    for entry in metrics.values():
        record = {
            "Strategy": entry.name,
            "TotalReturn": entry.total_return,
            "AnnReturn": entry.annualized_return,
            "AnnVol": entry.annualized_vol,
            "Sharpe": entry.sharpe,
            "MaxDrawdown": entry.max_drawdown,
            "Calmar": entry.calmar,
            "AvgTurnover": entry.avg_turnover,
        }
        record.update({f"Exposure_{k}": v for k, v in entry.realized_factor_exposure.items()})
        records.append(record)
    df = pd.DataFrame(records).set_index("Strategy")
    df.sort_values("Sharpe", ascending=False, inplace=True)
    return df


def persist_outputs(metrics: Dict[str, StrategyMetrics]) -> None:
    summary = summarise_metrics(metrics)
    summary_path = RESULTS_DIR / "evaluation_summary.csv"
    summary.to_csv(summary_path)

    md_path = RESULTS_DIR / "evaluation_summary.md"
    try:
        summary.round(4).to_markdown(md_path)
    except ImportError:
        # Fall back to plain text if optional tabulate dependency is missing
        md_path.write_text(summary.round(4).to_string())

    for name, entry in metrics.items():
        base = RESULTS_DIR / name
        base.mkdir(exist_ok=True)
        entry.returns.to_csv(base / "returns.csv", header=True)
        entry.weights.to_csv(base / "weights.csv")
        pd.Series(entry.realized_factor_exposure).to_csv(base / "factor_exposure.csv")


# ---------------------------------------------------------------------------
# CLI -----------------------------------------------------------------------
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the comprehensive evaluation")
    parser.add_argument(
        "--training-window",
        type=int,
        default=252,
        help="Training window length in trading days (legacy flag name retained)",
    )
    parser.add_argument(
        "--rebalance-frequency",
        type=int,
        default=1,
        help="Rebalance frequency in trading days (legacy flag name retained)",
    )
    parser.add_argument("--cost", type=float, default=0.001, help="Transaction cost per dollar traded")
    parser.add_argument("--smoothing", type=float, default=0.35, help="Portfolio smoothing parameter")
    parser.add_argument(
        "--grid-level",
        choices=tuple(GRID_LIBRARY.keys()),
        default="small",
        help="Size of the hyperparameter grid for conditional knockoff tuning",
    )
    parser.add_argument(
        "--search-mode",
        choices=("grid", "random"),
        default="grid",
        help="Hyperparameter search strategy",
    )
    parser.add_argument(
        "--search-samples",
        type=int,
        default=None,
        help="Number of random samples when using random search",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help=(
            "Comma-separated tickers when using yfinance data. "
            f"Defaults to {','.join(DEFAULT_TICKERS)}"
        ),
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2014-01-01",
        help="Start date for yfinance data",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="End date for yfinance data",
    )
    parser.add_argument(
        "--target-beta",
        type=float,
        default=None,
        help="Target beta for long-only portfolios (e.g., 1.0 for market-neutral, 0.8 for defensive). If not set, uses beta_neutral=True (requires shorting)",
    )
    parser.add_argument(
        "--beta-tolerance",
        type=float,
        default=0.15,
        help="Tolerance for target_beta constraint (default: 0.15)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    param_grid = GRID_LIBRARY[args.grid_level]
    if args.tickers:
        tickers = [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
    else:
        tickers = list(DEFAULT_TICKERS)

    metrics = run_evaluation(
        training_window_days=args.training_window,
        rebalance_frequency_days=args.rebalance_frequency,
        cost_per_dollar=args.cost,
        smoothing=args.smoothing,
        param_grid=param_grid,
        search_mode=args.search_mode,
        search_samples=args.search_samples,
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        target_beta=args.target_beta,
        beta_tolerance=args.beta_tolerance,
    )
    persist_outputs(metrics)

    summary = summarise_metrics(metrics)
    print("Evaluation complete. Summary:")
    print(summary.round(3))


if __name__ == "__main__":
    main()
