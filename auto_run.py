"""
自動化策略優化執行腳本
持續優化直到找到有效策略
"""
import sys
import argparse
import numpy as np
from datetime import datetime
from data_fetcher import TWStockDataFetcher
from auto_optimizer import AutoStrategyOptimizer
from strategy import MovingAverageStrategy, RSIStrategy, MACDStrategy, CombinedStrategy
from backtest_engine import BacktestEngine
from performance import PerformanceAnalyzer


def create_strategy_configs():
    """創建策略配置列表"""
    return [
        {
            'class': MovingAverageStrategy,
            'param_grid': {
                'short_window': [3, 5, 7, 10, 12, 15],
                'long_window': [15, 20, 25, 30, 35, 40, 50]
            },
            'method': 'grid_search',
            'name': '移動平均策略'
        },
        {
            'class': RSIStrategy,
            'param_ranges': {
                'period': (10, 25, 'int'),
                'oversold': (15, 35, 'float'),
                'overbought': (65, 85, 'float')
            },
            'method': 'random_search',
            'name': 'RSI 策略'
        },
        {
            'class': MACDStrategy,
            'param_ranges': {
                'fast': (8, 16, 'int'),
                'slow': (20, 35, 'int'),
                'signal': (5, 15, 'int')
            },
            'method': 'random_search',
            'name': 'MACD 策略'
        },
        {
            'class': CombinedStrategy,
            'param_ranges': {
                'ma_short': (3, 12, 'int'),
                'ma_long': (15, 35, 'int'),
                'rsi_period': (10, 20, 'int')
            },
            'method': 'random_search',
            'name': '組合策略'
        }
    ]


def auto_optimize(
    symbol: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 1000000,
    train_ratio: float = 0.7,
    optimization_metric: str = 'custom',
    max_combinations: int = 100,
    min_test_score: float = 0.0,
    max_iterations: int = 3
):
    """
    自動優化策略，持續嘗試直到找到有效策略
    
    Parameters:
    -----------
    symbol : str
        股票代碼
    start_date : str
        開始日期
    end_date : str
        結束日期
    initial_capital : float
        初始資金
    train_ratio : float
        訓練集比例
    optimization_metric : str
        優化目標
    max_combinations : int
        每個策略最大測試組合數
    min_test_score : float
        樣本外測試最低分數要求
    max_iterations : int
        最大迭代次數（如果找不到有效策略）
    """
    print("\n" + "="*70)
    print("自動化策略優化系統 - 持續優化模式")
    print("="*70)
    print(f"股票代碼: {symbol}")
    print(f"日期範圍: {start_date} 至 {end_date}")
    print(f"訓練集比例: {train_ratio*100:.0f}%")
    print(f"優化目標: {optimization_metric}")
    print(f"最低樣本外分數要求: {min_test_score}")
    print("="*70)
    
    # 獲取數據
    print("\n[1/4] 正在獲取股票數據...")
    fetcher = TWStockDataFetcher()
    try:
        data = fetcher.fetch_data(symbol, start_date, end_date)
        print(f"✓ 成功獲取 {len(data)} 筆數據")
        print(f"  日期範圍: {data.index[0]} 至 {data.index[-1]}")
    except Exception as e:
        print(f"✗ 錯誤: {str(e)}")
        return None
    
    # 創建優化器
    print("\n[2/4] 初始化優化器...")
    optimizer = AutoStrategyOptimizer(
        data=data,
        initial_capital=initial_capital,
        train_ratio=train_ratio,
        optimization_metric=optimization_metric,
        min_trades=5
    )
    print(f"✓ 訓練集: {len(optimizer.train_data)} 天")
    print(f"✓ 測試集: {len(optimizer.test_data)} 天")
    
    # 獲取策略配置
    strategies_config = create_strategy_configs()
    print(f"\n[3/4] 將測試 {len(strategies_config)} 種策略類型")
    
    # 執行優化
    print("\n[4/4] 開始優化...")
    best_strategy_info = None
    
    for iteration in range(max_iterations):
        print(f"\n--- 第 {iteration + 1} 輪優化 ---")
        
        result = optimizer.find_best_strategy(
            strategies_config,
            max_combinations_per_strategy=max_combinations
        )
        
        if result:
            test_score = result.get('test_score', -np.inf)
            test_result = result.get('test_result', {})
            
            if test_result.get('valid'):
                print(f"\n✓ 找到有效策略！")
                print(f"  樣本外分數: {test_score:.4f}")
                
                if test_score >= min_test_score:
                    print(f"✓ 樣本外分數達到要求 ({min_test_score})")
                    best_strategy_info = result
                    break
                else:
                    print(f"⚠ 樣本外分數未達要求，繼續優化...")
            else:
                print(f"⚠ 策略在樣本外數據上表現不佳，繼續優化...")
        else:
            print(f"⚠ 本輪未找到有效策略，繼續嘗試...")
        
        # 如果已經是最後一輪，使用當前最佳結果
        if iteration == max_iterations - 1 and result:
            print(f"\n已達最大迭代次數，使用當前最佳策略")
            best_strategy_info = result
    
    if not best_strategy_info:
        print("\n✗ 未能找到有效策略")
        return None
    
    # 在完整數據上評估
    print("\n" + "="*70)
    print("最終評估 - 在完整數據上測試最佳策略")
    print("="*70)
    
    full_result = optimizer.evaluate_on_full_data()
    
    if full_result and full_result.get('valid'):
        best_strategy = optimizer.get_best_strategy()
        
        engine = BacktestEngine(initial_capital=initial_capital)
        equity_df = engine.run(data, best_strategy)
        analyzer = PerformanceAnalyzer(equity_df, initial_capital)
        
        # 顯示完整報告
        print("\n" + "="*70)
        print("最終績效報告")
        print("="*70)
        
        report = analyzer.generate_report(
            strategy_name=f"{symbol} - {best_strategy_info.get('strategy_name')} (自動優化)"
        )
        print(report)
        
        # 顯示參數資訊
        print("\n" + "-"*70)
        print("策略參數資訊")
        print("-"*70)
        print(f"策略類型: {best_strategy_info.get('strategy_name')}")
        print(f"最佳參數: {best_strategy_info.get('params')}")
        print(f"樣本內分數: {best_strategy_info.get('score', 0):.4f}")
        print(f"樣本外分數: {best_strategy_info.get('test_score', 0):.4f}")
        print(f"完整數據總報酬: {full_result.get('total_return', 0)*100:.2f}%")
        print(f"完整數據夏普比率: {full_result.get('sharpe_ratio', 0):.2f}")
        print(f"完整數據最大回撤: {full_result.get('max_drawdown', 0)*100:.2f}%")
        print(f"勝率: {full_result.get('win_rate', 0)*100:.2f}%")
        print(f"總交易次數: {full_result.get('total_trades', 0)}")
        
        # 顯示交易記錄摘要
        trades_df = engine.get_trades()
        if not trades_df.empty:
            print(f"\n交易記錄摘要 (共 {len(trades_df)} 筆):")
            print(trades_df.head(10).to_string())
            if len(trades_df) > 10:
                print(f"... (還有 {len(trades_df) - 10} 筆)")
        
        # 繪製圖表
        print("\n正在生成視覺化圖表...")
        benchmark = data['close']
        analyzer.plot_equity_curve(benchmark=benchmark)
        analyzer.plot_drawdown()
        analyzer.plot_monthly_returns()
        
        return {
            'strategy': best_strategy,
            'equity_df': equity_df,
            'trades_df': trades_df,
            'analyzer': analyzer,
            'engine': engine,
            'best_strategy_info': best_strategy_info,
            'full_result': full_result
        }
    
    return None


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='自動化策略優化系統')
    parser.add_argument('--symbol', type=str, default='2330', help='股票代碼 (預設: 2330)')
    parser.add_argument('--start', type=str, default='2020-01-01', help='開始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2023-12-31', help='結束日期 (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=1000000, help='初始資金 (預設: 1000000)')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='訓練集比例 (預設: 0.7)')
    parser.add_argument('--metric', type=str, default='custom', 
                       choices=['sharpe_ratio', 'total_return', 'calmar_ratio', 'custom'],
                       help='優化目標指標 (預設: custom)')
    parser.add_argument('--max-combinations', type=int, default=100, 
                       help='每個策略最大測試組合數 (預設: 100)')
    parser.add_argument('--min-score', type=float, default=0.0, 
                       help='樣本外測試最低分數要求 (預設: 0.0)')
    parser.add_argument('--max-iterations', type=int, default=3, 
                       help='最大迭代次數 (預設: 3)')
    
    args = parser.parse_args()
    
    result = auto_optimize(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        train_ratio=args.train_ratio,
        optimization_metric=args.metric,
        max_combinations=args.max_combinations,
        min_test_score=args.min_score,
        max_iterations=args.max_iterations
    )
    
    if result:
        print("\n" + "="*70)
        print("✓ 優化完成！")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("✗ 優化失敗")
        print("="*70)
        sys.exit(1)


if __name__ == "__main__":
    main()

