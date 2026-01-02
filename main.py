"""
台灣股票回測系統主程式
"""
import pandas as pd
from datetime import datetime, timedelta
from data_fetcher import TWStockDataFetcher
from strategy import MovingAverageStrategy, RSIStrategy, MACDStrategy, CombinedStrategy
from backtest_engine import BacktestEngine
from performance import PerformanceAnalyzer
from auto_optimizer import AutoStrategyOptimizer


def run_backtest(
    symbol: str,
    strategy,
    start_date: str,
    end_date: str,
    initial_capital: float = 1000000,
    commission: float = 0.001425,
    tax: float = 0.003,
    slippage: float = 0.001,
    show_plots: bool = True,
    save_results: bool = False
):
    """
    執行回測
    
    Parameters:
    -----------
    symbol : str
        股票代碼（例如: '2330'）
    strategy : BaseStrategy
        交易策略
    start_date : str
        開始日期 (YYYY-MM-DD)
    end_date : str
        結束日期 (YYYY-MM-DD)
    initial_capital : float
        初始資金
    commission : float
        手續費率
    tax : float
        交易稅率（賣出時）
    slippage : float
        滑點
    show_plots : bool
        是否顯示圖表
    save_results : bool
        是否儲存結果
    """
    print(f"\n{'='*60}")
    print(f"開始回測: {symbol} - {strategy.name}")
    print(f"日期範圍: {start_date} 至 {end_date}")
    print(f"{'='*60}\n")
    
    # 獲取數據
    print("正在獲取股票數據...")
    fetcher = TWStockDataFetcher()
    try:
        data = fetcher.fetch_data(symbol, start_date, end_date)
        print(f"成功獲取 {len(data)} 筆數據\n")
    except Exception as e:
        print(f"錯誤: {str(e)}")
        return None
    
    # 執行回測
    print("正在執行回測...")
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=commission,
        tax=tax,
        slippage=slippage
    )
    
    equity_df = engine.run(data, strategy)
    
    # 績效分析
    print("正在分析績效...\n")
    analyzer = PerformanceAnalyzer(equity_df, initial_capital)
    
    # 顯示報告
    report = analyzer.generate_report(strategy_name=f"{symbol} - {strategy.name}")
    print(report)
    
    # 獲取交易記錄
    trades_df = engine.get_trades()
    if not trades_df.empty:
        print(f"\n交易記錄 (共 {len(trades_df)} 筆):")
        print(trades_df.to_string())
        print()
    
    # 繪製圖表
    if show_plots:
        print("正在生成圖表...")
        
        # 基準（買入持有）
        benchmark = data['close']
        
        analyzer.plot_equity_curve(benchmark=benchmark)
        analyzer.plot_drawdown()
        analyzer.plot_monthly_returns()
        analyzer.plot_returns_distribution()
    
    # 儲存結果
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"results_{symbol}_{strategy.name}_{timestamp}"
        
        equity_df.to_csv(f"{prefix}_equity.csv")
        if not trades_df.empty:
            trades_df.to_csv(f"{prefix}_trades.csv")
        
        with open(f"{prefix}_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n結果已儲存至: {prefix}_*")
    
    return {
        'equity_df': equity_df,
        'trades_df': trades_df,
        'analyzer': analyzer,
        'engine': engine
    }


def compare_strategies(
    symbol: str,
    strategies: list,
    start_date: str,
    end_date: str,
    initial_capital: float = 1000000
):
    """比較多個策略的績效"""
    print(f"\n{'='*60}")
    print(f"策略比較: {symbol}")
    print(f"日期範圍: {start_date} 至 {end_date}")
    print(f"{'='*60}\n")
    
    fetcher = TWStockDataFetcher()
    data = fetcher.fetch_data(symbol, start_date, end_date)
    
    results = {}
    
    for strategy in strategies:
        engine = BacktestEngine(initial_capital=initial_capital)
        equity_df = engine.run(data, strategy)
        analyzer = PerformanceAnalyzer(equity_df, initial_capital)
        
        results[strategy.name] = {
            'equity_df': equity_df,
            'analyzer': analyzer,
            'metrics': analyzer.calculate_metrics()
        }
        
        print(f"\n{strategy.name}:")
        print(f"  總報酬率: {analyzer.calculate_metrics().get('總報酬率', 'N/A')}")
        print(f"  夏普比率: {analyzer.calculate_metrics().get('夏普比率', 'N/A')}")
        print(f"  最大回撤: {analyzer.calculate_metrics().get('最大回撤', 'N/A')}")
    
    # 繪製比較圖
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for name, result in results.items():
        equity_df = result['equity_df']
        ax.plot(equity_df.index, equity_df['total_equity'], 
                label=name, linewidth=2)
    
    # 基準
    benchmark = data['close'] / data['close'].iloc[0] * initial_capital
    ax.plot(benchmark.index, benchmark, 
            label='買入持有', linewidth=2, linestyle='--', color='gray')
    
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('權益 (NTD)', fontsize=12)
    ax.set_title(f'{symbol} 策略比較', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results


def auto_optimize_strategy(
    symbol: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 1000000,
    train_ratio: float = 0.7,
    optimization_metric: str = 'custom',
    show_plots: bool = True,
    save_results: bool = False
):
    """
    自動優化策略，找到最佳策略
    
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
    show_plots : bool
        是否顯示圖表
    save_results : bool
        是否儲存結果
    """
    print("\n" + "="*60)
    print("自動化策略優化系統")
    print("="*60)
    print(f"股票代碼: {symbol}")
    print(f"日期範圍: {start_date} 至 {end_date}")
    print(f"優化目標: {optimization_metric}")
    print("="*60)
    
    # 獲取數據
    print("\n正在獲取股票數據...")
    fetcher = TWStockDataFetcher()
    try:
        data = fetcher.fetch_data(symbol, start_date, end_date)
        print(f"成功獲取 {len(data)} 筆數據")
    except Exception as e:
        print(f"錯誤: {str(e)}")
        return None
    
    # 創建自動優化器
    optimizer = AutoStrategyOptimizer(
        data=data,
        initial_capital=initial_capital,
        train_ratio=train_ratio,
        optimization_metric=optimization_metric,
        min_trades=5
    )
    
    # 定義策略配置
    strategies_config = [
        {
            'class': MovingAverageStrategy,
            'param_grid': {
                'short_window': [3, 5, 7, 10, 12],
                'long_window': [15, 20, 25, 30, 35]
            },
            'method': 'grid_search'
        },
        {
            'class': RSIStrategy,
            'param_ranges': {
                'period': (10, 20, 'int'),
                'oversold': (20, 40, 'float'),
                'overbought': (60, 80, 'float')
            },
            'method': 'random_search'
        },
        {
            'class': MACDStrategy,
            'param_ranges': {
                'fast': (8, 15, 'int'),
                'slow': (20, 30, 'int'),
                'signal': (5, 12, 'int')
            },
            'method': 'random_search'
        },
        {
            'class': CombinedStrategy,
            'param_ranges': {
                'ma_short': (3, 10, 'int'),
                'ma_long': (15, 30, 'int'),
                'rsi_period': (10, 18, 'int')
            },
            'method': 'random_search'
        }
    ]
    
    # 執行自動優化
    best_strategy_info = optimizer.find_best_strategy(
        strategies_config,
        max_combinations_per_strategy=50
    )
    
    if not best_strategy_info:
        print("\n未能找到有效策略")
        return None
    
    # 在完整數據上評估
    full_result = optimizer.evaluate_on_full_data()
    
    if full_result and full_result.get('valid'):
        # 使用最佳策略執行完整回測
        best_strategy = optimizer.get_best_strategy()
        
        print("\n" + "="*60)
        print("使用最佳策略執行完整回測")
        print("="*60)
        
        engine = BacktestEngine(initial_capital=initial_capital)
        equity_df = engine.run(data, best_strategy)
        analyzer = PerformanceAnalyzer(equity_df, initial_capital)
        
        # 顯示報告
        report = analyzer.generate_report(
            strategy_name=f"{symbol} - {best_strategy_info.get('strategy_name')} (優化後)"
        )
        print(report)
        
        # 顯示交易記錄
        trades_df = engine.get_trades()
        if not trades_df.empty:
            print(f"\n交易記錄 (共 {len(trades_df)} 筆):")
            print(trades_df.head(20).to_string())
            if len(trades_df) > 20:
                print(f"... (還有 {len(trades_df) - 20} 筆)")
        
        # 繪製圖表
        if show_plots:
            print("\n正在生成圖表...")
            benchmark = data['close']
            analyzer.plot_equity_curve(benchmark=benchmark)
            analyzer.plot_drawdown()
            analyzer.plot_monthly_returns()
        
        # 儲存結果
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = f"optimized_{symbol}_{timestamp}"
            
            equity_df.to_csv(f"{prefix}_equity.csv")
            if not trades_df.empty:
                trades_df.to_csv(f"{prefix}_trades.csv")
            
            with open(f"{prefix}_report.txt", 'w', encoding='utf-8') as f:
                f.write(report)
                f.write(f"\n最佳參數: {best_strategy_info.get('params')}\n")
                f.write(f"樣本內分數: {best_strategy_info.get('score', 0):.4f}\n")
                f.write(f"樣本外分數: {best_strategy_info.get('test_score', 0):.4f}\n")
            
            print(f"\n結果已儲存至: {prefix}_*")
        
        return {
            'strategy': best_strategy,
            'equity_df': equity_df,
            'trades_df': trades_df,
            'analyzer': analyzer,
            'engine': engine,
            'best_strategy_info': best_strategy_info
        }
    
    return None


if __name__ == "__main__":
    """
    主程式入口
    
    使用方式:
    - python main.py                    # 使用預設參數執行自動優化
    - python main.py --auto 2330        # 指定股票代碼
    - python main.py --auto 2330 2020-01-01 2023-12-31  # 完整參數
    """
    import sys
    
    # 預設參數
    symbol = "2330"
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    
    # 解析命令列參數
    if len(sys.argv) > 1:
        if sys.argv[1] == '--auto':
            symbol = sys.argv[2] if len(sys.argv) > 2 else symbol
            start_date = sys.argv[3] if len(sys.argv) > 3 else start_date
            end_date = sys.argv[4] if len(sys.argv) > 4 else end_date
        elif sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print(__doc__)
            print("\n建議使用 auto_run.py 進行自動化優化:")
            print("  python auto_run.py --help")
            sys.exit(0)
    
    # 執行自動優化
    result = auto_optimize_strategy(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        initial_capital=1000000,
        train_ratio=0.7,
        optimization_metric='custom',
        show_plots=True,
        save_results=False
    )

