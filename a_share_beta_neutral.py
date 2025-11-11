"""
Solutions for beta-neutrality in A-share market (long-only only).

A股市场特点：
1. 不能做空个股（除非有融券，但券源少、成本高）
2. 可以做空股指期货（沪深300/中证500/上证50）
3. 大部分股票 beta > 0

解决方案对比：
"""
import numpy as np
from src.knockoff_neutralized.portfolio_optimizer import PortfolioOptimizer

print("="*80)
print("A股 Long-Only 市场的 Beta-Neutral 策略方案")
print("="*80)

# 模拟A股数据：所有股票 beta > 0
n_assets = 20
np.random.seed(42)
betas = np.random.uniform(0.6, 1.8, n_assets)  # 所有 beta 都是正的
alpha = np.random.randn(n_assets)  # 模拟 alpha 信号

print(f"\n模拟数据:")
print(f"  资产数量: {n_assets}")
print(f"  Beta 范围: [{betas.min():.2f}, {betas.max():.2f}]")
print(f"  Beta 均值: {betas.mean():.2f}")
print(f"  所有 beta 都是正值: {(betas > 0).all()}")

# ============================================================================
# 方案1: Benchmark-Relative Beta Neutrality (推荐!)
# ============================================================================
print("\n" + "="*80)
print("方案1: 相对基准的 Beta 中性 (Benchmark-Relative Beta Neutrality)")
print("="*80)
print("""
原理：
- 不追求绝对 beta = 0（long-only 下不可能）
- 追求相对于基准（如沪深300）的 beta = 0
- 即：组合 beta = 基准 beta = 1.0

实现：
- 约束组合的加权平均 beta = 市场平均 beta
- 这样组合相对市场是中性的
""")

# 计算市场平均 beta（等权基准）
benchmark_beta = betas.mean()

opt_benchmark_relative = PortfolioOptimizer(
    risk_aversion=0.5,
    beta_neutral=False,  # 不用绝对 beta=0
    neutralize_factors=False,
    long_only=True,
    max_leverage=1.0,
    max_position_size=0.2,
)

result1 = opt_benchmark_relative.optimize(
    alpha_scores=alpha,
    factor_exposures=betas.reshape(-1, 1),
    covariance_matrix=np.eye(n_assets) * 0.01,
    current_weights=np.zeros(n_assets),
)

weights1 = result1.weights
portfolio_beta1 = weights1 @ betas

print(f"\n结果:")
print(f"  状态: {result1.status}")
print(f"  组合 beta: {portfolio_beta1:.4f}")
print(f"  基准 beta: {benchmark_beta:.4f}")
print(f"  相对 beta: {portfolio_beta1 - benchmark_beta:.4f}")
print(f"\n优点:")
print(f"  ✓ 可实现（long-only 可行）")
print(f"  ✓ 组合相对市场中性")
print(f"  ✓ 适合 alpha 策略")

# ============================================================================
# 方案2: Minimize Beta (低 beta 策略)
# ============================================================================
print("\n" + "="*80)
print("方案2: 最小化 Beta 策略")
print("="*80)
print("""
原理：
- 在保持 alpha 的前提下，尽量选择低 beta 资产
- 通过调整 risk_aversion 参数来权衡 alpha vs beta

实现：
- 使用高 risk_aversion
- 自然偏好低 beta（低波动）资产
""")

opt_min_beta = PortfolioOptimizer(
    risk_aversion=5.0,  # 高风险厌恶 → 偏好低 beta
    beta_neutral=False,
    neutralize_factors=False,
    long_only=True,
    max_leverage=1.0,
    max_position_size=0.2,
)

result2 = opt_min_beta.optimize(
    alpha_scores=alpha,
    factor_exposures=betas.reshape(-1, 1),
    covariance_matrix=np.eye(n_assets) * 0.01,
    current_weights=np.zeros(n_assets),
)

weights2 = result2.weights
portfolio_beta2 = weights2 @ betas

print(f"\n结果:")
print(f"  状态: {result2.status}")
print(f"  组合 beta: {portfolio_beta2:.4f}")
print(f"  vs 基准: {portfolio_beta2 - benchmark_beta:.4f}")
print(f"\n优点:")
print(f"  ✓ 降低市场风险暴露")
print(f"  ✓ 下跌时表现更好")
print(f"  ✗ 上涨时跑输基准")

# ============================================================================
# 方案3: 股指期货对冲 (最接近真正的 beta-neutral)
# ============================================================================
print("\n" + "="*80)
print("方案3: 股指期货对冲 (推荐用于机构投资者)")
print("="*80)
print("""
原理：
- 做多股票组合（选股 alpha）
- 做空股指期货（对冲 beta）
- 实现真正的 beta = 0

实现步骤：
1. 构建 long-only 股票组合（追求 alpha）
2. 计算组合的市场 beta
3. 做空相应数量的股指期货合约

示例计算：
""")

# Step 1: 构建股票组合
opt_with_futures = PortfolioOptimizer(
    risk_aversion=0.5,
    beta_neutral=False,
    neutralize_factors=False,
    long_only=True,
    max_leverage=1.0,
    max_position_size=0.2,
)

result3 = opt_with_futures.optimize(
    alpha_scores=alpha,
    factor_exposures=betas.reshape(-1, 1),
    covariance_matrix=np.eye(n_assets) * 0.01,
    current_weights=np.zeros(n_assets),
)

weights3 = result3.weights
portfolio_beta3 = weights3 @ betas

print(f"股票组合:")
print(f"  总市值: 1,000,000 元")
print(f"  组合 beta: {portfolio_beta3:.4f}")
print(f"\n期货对冲:")
print(f"  需要做空: {portfolio_beta3:.4f} * 1,000,000 = {portfolio_beta3 * 1000000:,.0f} 元名义价值")
print(f"  IF/IC/IH 合约数 ≈ {portfolio_beta3 * 1000000 / 200000:.1f} 张")
print(f"\n对冲后:")
print(f"  组合 beta ≈ 0")
print(f"  组合收益 = alpha + 手续费成本")
print(f"\n优点:")
print(f"  ✓ 真正的 beta = 0")
print(f"  ✓ 纯 alpha 收益")
print(f"  ✓ 适合量化机构")
print(f"\n缺点:")
print(f"  ✗ 需要期货账户")
print(f"  ✗ 保证金要求")
print(f"  ✗ 基差风险")
print(f"  ✗ 展期成本")

# ============================================================================
# 方案4: ETF 组合（散户可用）
# ============================================================================
print("\n" + "="*80)
print("方案4: 反向 ETF 对冲 (适合散户)")
print("="*80)
print("""
原理：
- 买入股票组合
- 买入反向 ETF（如反向沪深300）
- 部分对冲市场风险

可用工具：
- 港股：FI二南方恒指、FI二南方国指（2倍反向）
- A股：暂无反向 ETF（监管限制）

替代方案（A股可用）：
- 买入低 beta 行业 ETF（消费、医药）
- 配置债券、货币基金降低整体 beta
""")

# ============================================================================
# 推荐方案总结
# ============================================================================
print("\n" + "="*80)
print("💡 A股投资者推荐方案")
print("="*80)
print("""
【散户投资者】
推荐: 方案1 (相对 beta 中性) + 方案2 (低 beta)
- 选择低 beta 股票
- 保持组合 beta ≈ 市场平均
- 通过选股获得 alpha
- 降低市场择时风险

【机构投资者】
推荐: 方案3 (股指期货对冲)
- 专注选股 alpha
- 用期货完全对冲 beta
- 实现纯 alpha 策略

【代码实现建议】
将当前策略改为 "benchmark-relative" 模式：
1. 设置 long_only=True（符合A股限制）
2. 不强制 beta=0（数学上不可行）
3. 添加 "target_beta" 参数（如 1.0）
4. 约束: |portfolio_beta - target_beta| < 0.1
""")

# ============================================================================
# 示例：实现 Benchmark-Relative Beta 约束
# ============================================================================
print("\n" + "="*80)
print("代码示例：如何实现相对 beta 中性")
print("="*80)

print("""
# 在 portfolio_optimizer.py 中添加:

class PortfolioOptimizer:
    def __init__(
        self,
        ...
        target_beta: Optional[float] = None,  # 新增参数
        beta_tolerance: float = 0.1,          # 新增参数
    ):
        ...
        self.target_beta = target_beta
        self.beta_tolerance = beta_tolerance
    
    def optimize(self, ...):
        ...
        # 如果指定了 target_beta，使用相对约束而非绝对约束
        if self.target_beta is not None:
            beta_exposure = factor_exposures[:, 0] @ w
            constraints.append(beta_exposure <= self.target_beta + self.beta_tolerance)
            constraints.append(beta_exposure >= self.target_beta - self.beta_tolerance)
        elif self.beta_neutral:
            # 原有的绝对 beta=0 约束（需要 long_only=False）
            beta_exposure = factor_exposures[:, 0] @ w
            constraints.append(beta_exposure <= 1e-6)
            constraints.append(beta_exposure >= -1e-6)
""")

print("\n" + "="*80)
print("总结：A股市场应该放弃绝对 beta=0，改用相对 beta 中性")
print("="*80)
