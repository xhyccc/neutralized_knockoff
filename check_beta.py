import sys
sys.path.insert(0, 'examples')
from comprehensive_evaluation import run_evaluation, summarise_metrics

metrics = run_evaluation(
    search_mode='random',
    search_samples=4,
    start_date='2022-01-01',
    end_date='2025-10-30'
)
summary = summarise_metrics(metrics)
print('\n=== Key Metrics (focus on MarketBeta exposure) ===')
cols = ['TotalReturn', 'Sharpe', 'Exposure_MarketBeta']
print(summary[cols].round(4).to_string())
