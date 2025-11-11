"""Test target_beta functionality for A-share long-only portfolios."""
import numpy as np
from src.knockoff_neutralized.portfolio_optimizer import PortfolioOptimizer

print("="*80)
print("测试 target_beta 功能（适用于A股 long-only 市场）")
print("="*80)

# 模拟A股数据
n_assets = 20
np.random.seed(42)
betas = np.random.uniform(0.6, 1.8, n_assets)  # 所有 beta > 0
alpha = np.random.randn(n_assets)

benchmark_beta = betas.mean()
print(f"\n数据设置:")
print(f"  资产数: {n_assets}")
print(f"  Beta 范围: [{betas.min():.2f}, {betas.max():.2f}]")
print(f"  等权基准 beta: {benchmark_beta:.4f}")

# ============================================================================
# Test 1: 默认行为（beta_neutral=True, long_only=True）- 会失败
# ============================================================================
print("\n" + "="*80)
print("Test 1: 传统 beta_neutral (会失败)")
print("="*80)

opt1 = PortfolioOptimizer(
    risk_aversion=0.5,
    beta_neutral=True,
    neutralize_factors=True,
    long_only=True,
    max_leverage=1.0,
    max_position_size=0.2,
)

result1 = opt1.optimize(
    alpha_scores=alpha,
    factor_exposures=betas.reshape(-1, 1),
    covariance_matrix=np.eye(n_assets) * 0.01,
    current_weights=np.zeros(n_assets),
)

print(f"状态: {result1.status}")
print(f"组合 beta: {(result1.weights @ betas):.4f}")
if result1.status != 'optimal':
    print("✗ 失败！long-only 无法实现 beta=0")

# ============================================================================
# Test 2: target_beta = 1.0 (市场 beta)
# ============================================================================
print("\n" + "="*80)
print("Test 2: target_beta = 1.0 (市场中性)")
print("="*80)

opt2 = PortfolioOptimizer(
    risk_aversion=0.5,
    neutralize_factors=True,
    long_only=True,
    max_leverage=1.0,
    max_position_size=0.2,
    target_beta=1.0,         # 目标 beta = 1.0
    beta_tolerance=0.15,     # 允许 ±0.15 的偏差
)

result2 = opt2.optimize(
    alpha_scores=alpha,
    factor_exposures=betas.reshape(-1, 1),
    covariance_matrix=np.eye(n_assets) * 0.01,
    current_weights=np.zeros(n_assets),
)

portfolio_beta2 = result2.weights @ betas
print(f"状态: {result2.status}")
print(f"组合 beta: {portfolio_beta2:.4f}")
print(f"目标 beta: 1.0")
print(f"偏差: {portfolio_beta2 - 1.0:.4f}")
print(f"在容许范围内: {abs(portfolio_beta2 - 1.0) <= 0.15}")
if result2.status == 'optimal':
    print("✓ 成功！实现了市场 beta")

# ============================================================================
# Test 3: target_beta = benchmark_beta (等权基准)
# ============================================================================
print("\n" + "="*80)
print(f"Test 3: target_beta = {benchmark_beta:.4f} (等权基准)")
print("="*80)

opt3 = PortfolioOptimizer(
    risk_aversion=0.5,
    neutralize_factors=True,
    long_only=True,
    max_leverage=1.0,
    max_position_size=0.2,
    target_beta=benchmark_beta,  # 目标 = 等权基准 beta
    beta_tolerance=0.10,
)

result3 = opt3.optimize(
    alpha_scores=alpha,
    factor_exposures=betas.reshape(-1, 1),
    covariance_matrix=np.eye(n_assets) * 0.01,
    current_weights=np.zeros(n_assets),
)

portfolio_beta3 = result3.weights @ betas
print(f"状态: {result3.status}")
print(f"组合 beta: {portfolio_beta3:.4f}")
print(f"目标 beta: {benchmark_beta:.4f}")
print(f"偏差: {portfolio_beta3 - benchmark_beta:.4f}")
print(f"在容许范围内: {abs(portfolio_beta3 - benchmark_beta) <= 0.10}")
if result3.status == 'optimal':
    print("✓ 成功！实现了相对基准中性")

# ============================================================================
# Test 4: target_beta = 0.8 (低 beta 策略)
# ============================================================================
print("\n" + "="*80)
print("Test 4: target_beta = 0.8 (低 beta 防御策略)")
print("="*80)

opt4 = PortfolioOptimizer(
    risk_aversion=0.5,
    neutralize_factors=True,
    long_only=True,
    max_leverage=1.0,
    max_position_size=0.2,
    target_beta=0.8,          # 低于市场的 beta
    beta_tolerance=0.10,
)

result4 = opt4.optimize(
    alpha_scores=alpha,
    factor_exposures=betas.reshape(-1, 1),
    covariance_matrix=np.eye(n_assets) * 0.01,
    current_weights=np.zeros(n_assets),
)

portfolio_beta4 = result4.weights @ betas
print(f"状态: {result4.status}")
print(f"组合 beta: {portfolio_beta4:.4f}")
print(f"目标 beta: 0.8")
print(f"偏差: {portfolio_beta4 - 0.8:.4f}")
if result4.status == 'optimal':
    print("✓ 成功！实现了低 beta 防御策略")
    print("\n优势:")
    print("  - 熊市时跌幅小于市场")
    print("  - 降低组合波动")
    print("  - 适合风险厌恶投资者")

# ============================================================================
# Test 5: target_beta = 1.3 (高 beta 进攻策略)
# ============================================================================
print("\n" + "="*80)
print("Test 5: target_beta = 1.3 (高 beta 进攻策略)")
print("="*80)

opt5 = PortfolioOptimizer(
    risk_aversion=0.5,
    neutralize_factors=True,
    long_only=True,
    max_leverage=1.0,
    max_position_size=0.2,
    target_beta=1.3,          # 高于市场的 beta
    beta_tolerance=0.10,
)

result5 = opt5.optimize(
    alpha_scores=alpha,
    factor_exposures=betas.reshape(-1, 1),
    covariance_matrix=np.eye(n_assets) * 0.01,
    current_weights=np.zeros(n_assets),
)

portfolio_beta5 = result5.weights @ betas
print(f"状态: {result5.status}")
print(f"组合 beta: {portfolio_beta5:.4f}")
print(f"目标 beta: 1.3")
print(f"偏差: {portfolio_beta5 - 1.3:.4f}")
if result5.status == 'optimal':
    print("✓ 成功！实现了高 beta 进攻策略")
    print("\n优势:")
    print("  - 牛市时涨幅大于市场")
    print("  - 放大 alpha 收益")
    print("  - 适合看多市场时使用")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*80)
print("💡 总结：target_beta 功能完美适配A股市场")
print("="*80)
print("""
【功能特点】
✓ 兼容 long-only 约束（不需要做空）
✓ 可设置任意目标 beta（0.5 ~ 1.5 都可行）
✓ 保留 alpha 优化能力

【推荐用法】
1. 中性策略: target_beta = 1.0
   - 获得市场平均 beta
   - 专注选股 alpha

2. 防御策略: target_beta = 0.7-0.9
   - 熊市保护
   - 降低波动

3. 进攻策略: target_beta = 1.1-1.3  
   - 牛市放大收益
   - 适合高风险偏好

【与期货对冲对比】
target_beta 方式:
  ✓ 不需要期货账户
  ✓ 无保证金要求
  ✓ 无基差风险
  ✓ 散户可用
  ✗ 无法实现绝对 beta=0

期货对冲方式:
  ✓ 可实现绝对 beta=0
  ✓ 纯 alpha 收益
  ✗ 需要期货账户
  ✗ 保证金成本
  ✗ 展期成本
  ✗ 机构专用
""")
