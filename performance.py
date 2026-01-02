"""
績效分析模組
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
from datetime import datetime

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class PerformanceAnalyzer:
    """績效分析器"""
    
    def __init__(self, equity_df: pd.DataFrame, initial_capital: float):
        self.equity_df = equity_df
        self.initial_capital = initial_capital
        self.returns = equity_df['total_equity'].pct_change().dropna()
    
    def calculate_metrics(self) -> Dict:
        """計算績效指標"""
        if self.equity_df.empty:
            return {}
        
        final_equity = self.equity_df['total_equity'].iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # 年化報酬率
        days = (self.equity_df.index[-1] - self.equity_df.index[0]).days
        years = days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # 波動率
        volatility = self.returns.std() * np.sqrt(252) if len(self.returns) > 0 else 0
        
        # 夏普比率
        sharpe_ratio = np.sqrt(252) * self.returns.mean() / self.returns.std() if self.returns.std() > 0 else 0
        
        # 最大回撤
        max_drawdown = self._calculate_max_drawdown()
        
        # 卡爾瑪比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 勝率（需要交易記錄）
        win_rate = None
        
        return {
            '總報酬率': f"{total_return * 100:.2f}%",
            '年化報酬率': f"{annual_return * 100:.2f}%",
            '波動率': f"{volatility * 100:.2f}%",
            '夏普比率': f"{sharpe_ratio:.2f}",
            '最大回撤': f"{max_drawdown * 100:.2f}%",
            '卡爾瑪比率': f"{calmar_ratio:.2f}",
            '初始資金': f"${self.initial_capital:,.0f}",
            '最終權益': f"${final_equity:,.0f}",
            '交易天數': days
        }
    
    def _calculate_max_drawdown(self) -> float:
        """計算最大回撤"""
        cumulative = (1 + self.equity_df['total_equity'].pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def plot_equity_curve(self, benchmark: Optional[pd.Series] = None, save_path: Optional[str] = None):
        """繪製權益曲線"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 繪製策略權益曲線
        ax.plot(self.equity_df.index, self.equity_df['total_equity'], 
                label='策略權益', linewidth=2, color='#2E86AB')
        
        # 繪製基準（如果提供）
        if benchmark is not None:
            benchmark_normalized = benchmark / benchmark.iloc[0] * self.initial_capital
            ax.plot(benchmark.index, benchmark_normalized, 
                    label='基準（買入持有）', linewidth=2, linestyle='--', color='#A23B72')
        
        # 繪製初始資金線
        ax.axhline(y=self.initial_capital, color='gray', linestyle=':', alpha=0.5, label='初始資金')
        
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('權益 (NTD)', fontsize=12)
        ax.set_title('權益曲線', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_drawdown(self, save_path: Optional[str] = None):
        """繪製回撤圖"""
        cumulative = (1 + self.equity_df['total_equity'].pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        ax.plot(drawdown.index, drawdown, color='red', linewidth=1.5)
        
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('回撤 (%)', fontsize=12)
        ax.set_title('回撤分析', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_monthly_returns(self, save_path: Optional[str] = None):
        """繪製月度報酬率熱力圖"""
        monthly_returns = self.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        
        # 建立年度-月份矩陣
        returns_matrix = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first()
        returns_matrix = returns_matrix.unstack(level=1)
        returns_matrix.columns = ['1月', '2月', '3月', '4月', '5月', '6月', 
                                  '7月', '8月', '9月', '10月', '11月', '12月']
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(returns_matrix) * 0.5)))
        sns.heatmap(returns_matrix * 100, annot=True, fmt='.2f', cmap='RdYlGn', 
                    center=0, cbar_kws={'label': '報酬率 (%)'}, ax=ax)
        ax.set_title('月度報酬率熱力圖', fontsize=14, fontweight='bold')
        ax.set_xlabel('月份', fontsize=12)
        ax.set_ylabel('年份', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_returns_distribution(self, save_path: Optional[str] = None):
        """繪製報酬率分布"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 直方圖
        ax1.hist(self.returns * 100, bins=50, alpha=0.7, color='#2E86AB', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('日報酬率 (%)', fontsize=12)
        ax1.set_ylabel('頻率', fontsize=12)
        ax1.set_title('報酬率分布', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Q-Q 圖
        from scipy import stats
        stats.probplot(self.returns * 100, dist="norm", plot=ax2)
        ax2.set_title('Q-Q 圖（正態性檢驗）', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def generate_report(self, strategy_name: str = "策略", save_path: Optional[str] = None) -> str:
        """生成績效報告"""
        metrics = self.calculate_metrics()
        
        report = f"""
{'='*60}
{strategy_name} 回測績效報告
{'='*60}

基本資訊:
  - 初始資金: {metrics.get('初始資金', 'N/A')}
  - 最終權益: {metrics.get('最終權益', 'N/A')}
  - 交易天數: {metrics.get('交易天數', 'N/A')} 天

績效指標:
  - 總報酬率: {metrics.get('總報酬率', 'N/A')}
  - 年化報酬率: {metrics.get('年化報酬率', 'N/A')}
  - 波動率: {metrics.get('波動率', 'N/A')}
  - 夏普比率: {metrics.get('夏普比率', 'N/A')}
  - 最大回撤: {metrics.get('最大回撤', 'N/A')}
  - 卡爾瑪比率: {metrics.get('卡爾瑪比率', 'N/A')}

{'='*60}
"""
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report

