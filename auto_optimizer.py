"""
自動化策略優化系統
實現樣本內/樣本外測試，避免過度擬合
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from itertools import product
import random
from backtest_engine import BacktestEngine
from performance import PerformanceAnalyzer
from strategy import (
    BaseStrategy, MovingAverageStrategy, RSIStrategy, 
    MACDStrategy, CombinedStrategy
)
import warnings
warnings.filterwarnings('ignore')


class AutoStrategyOptimizer:
    """自動化策略優化器，包含樣本內/樣本外測試"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 1000000,
        train_ratio: float = 0.7,
        optimization_metric: str = 'sharpe_ratio',
        min_trades: int = 5,
        engine_params: Optional[Dict] = None
    ):
        """
        初始化自動優化器
        
        Parameters:
        -----------
        data : pd.DataFrame
            完整的回測數據
        initial_capital : float
            初始資金
        train_ratio : float
            訓練集比例（用於樣本內優化）
        optimization_metric : str
            優化目標指標
        min_trades : int
            最少交易次數（過濾無效策略）
        engine_params : dict, optional
            回測引擎參數
        """
        self.data = data
        self.initial_capital = initial_capital
        self.train_ratio = train_ratio
        self.optimization_metric = optimization_metric
        self.min_trades = min_trades
        self.engine_params = engine_params or {}
        
        # 分割數據
        split_idx = int(len(data) * train_ratio)
        self.train_data = data.iloc[:split_idx].copy()
        self.test_data = data.iloc[split_idx:].copy()
        
        self.results = []
        self.best_strategy_info = None
    
    def _evaluate_strategy(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        return_full: bool = False
    ) -> Dict:
        """
        評估策略績效
        
        Parameters:
        -----------
        strategy : BaseStrategy
            策略實例
        data : pd.DataFrame
            回測數據
        return_full : bool
            是否返回完整資訊
        
        Returns:
        --------
        dict
            績效指標
        """
        try:
            engine = BacktestEngine(
                initial_capital=self.initial_capital,
                **self.engine_params
            )
            equity_df = engine.run(data, strategy)
            
            analyzer = PerformanceAnalyzer(equity_df, self.initial_capital)
            stats = engine.get_final_stats()
            metrics = analyzer.calculate_metrics()
            
            # 檢查最少交易次數
            if stats.get('total_trades', 0) < self.min_trades:
                return {'valid': False, 'reason': '交易次數不足'}
            
            # 計算綜合評分
            total_return = stats.get('total_return', 0)
            sharpe = stats.get('sharpe_ratio', 0)
            max_dd = abs(float(str(metrics.get('最大回撤', '0%')).replace('%', '').replace('$', '').strip()) / 100)
            win_rate = stats.get('win_rate', 0)
            
            # 綜合評分：考慮報酬率、風險調整報酬、勝率
            if self.optimization_metric == 'sharpe_ratio':
                score = sharpe
            elif self.optimization_metric == 'total_return':
                score = total_return
            elif self.optimization_metric == 'custom':
                # 自訂評分：報酬率 * 夏普比率 * 勝率 / (1 + 最大回撤)
                score = total_return * sharpe * win_rate / (1 + max_dd) if max_dd > 0 else total_return * sharpe * win_rate
            else:
                score = sharpe
            
            result = {
                'valid': True,
                'score': score,
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'win_rate': win_rate,
                'total_trades': stats.get('total_trades', 0),
                'stats': stats,
                'metrics': metrics
            }
            
            if return_full:
                result['equity_df'] = equity_df
                result['engine'] = engine
                result['analyzer'] = analyzer
            
            return result
            
        except Exception as e:
            return {'valid': False, 'reason': str(e)}
    
    def optimize_strategy(
        self,
        strategy_class: type,
        param_grid: Optional[Dict] = None,
        param_ranges: Optional[Dict] = None,
        method: str = 'grid_search',
        max_combinations: int = 100
    ) -> Dict:
        """
        優化單一策略
        
        Parameters:
        -----------
        strategy_class : type
            策略類別
        param_grid : dict, optional
            參數網格（用於網格搜索）
        param_ranges : dict, optional
            參數範圍（用於隨機搜索）
        method : str
            優化方法 ('grid_search', 'random_search')
        max_combinations : int
            最大測試組合數（避免過度搜索）
        
        Returns:
        --------
        dict
            最佳策略資訊
        """
        print(f"\n{'='*60}")
        print(f"優化策略: {strategy_class.__name__}")
        print(f"{'='*60}")
        
        best_result = None
        best_score = -np.inf
        best_params = None
        tested_combinations = 0
        
        if method == 'grid_search' and param_grid:
            # 網格搜索
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            total_combinations = np.prod([len(v) for v in param_values])
            
            # 限制組合數
            if total_combinations > max_combinations:
                print(f"參數組合過多 ({total_combinations})，改用隨機搜索")
                method = 'random_search'
                # 轉換為範圍
                param_ranges = {}
                for name, values in param_grid.items():
                    if all(isinstance(v, int) for v in values):
                        param_ranges[name] = (min(values), max(values), 'int')
                    else:
                        param_ranges[name] = (min(values), max(values), 'float')
        
        if method == 'grid_search' and param_grid:
            # 執行網格搜索
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            
            for combination in product(*param_values):
                if tested_combinations >= max_combinations:
                    break
                
                params = dict(zip(param_names, combination))
                strategy = strategy_class(**params)
                
                result = self._evaluate_strategy(strategy, self.train_data)
                
                if result.get('valid') and result.get('score', -np.inf) > best_score:
                    best_score = result['score']
                    best_params = params.copy()
                    best_result = result.copy()
                    best_result['strategy'] = strategy
                    best_result['params'] = params
                
                tested_combinations += 1
                
                if tested_combinations % 10 == 0:
                    print(f"  已測試 {tested_combinations} 個組合，當前最佳分數: {best_score:.4f}")
        
        elif method == 'random_search' and param_ranges:
            # 執行隨機搜索
            n_iter = min(max_combinations, 100)
            
            for i in range(n_iter):
                params = {}
                for param_name, (min_val, max_val, param_type) in param_ranges.items():
                    if param_type == 'int':
                        params[param_name] = random.randint(int(min_val), int(max_val))
                    else:
                        params[param_name] = random.uniform(min_val, max_val)
                
                strategy = strategy_class(**params)
                result = self._evaluate_strategy(strategy, self.train_data)
                
                if result.get('valid') and result.get('score', -np.inf) > best_score:
                    best_score = result['score']
                    best_params = params.copy()
                    best_result = result.copy()
                    best_result['strategy'] = strategy
                    best_result['params'] = params
                
                if (i + 1) % 10 == 0:
                    print(f"  已測試 {i+1}/{n_iter} 個組合，當前最佳分數: {best_score:.4f}")
        
        if best_result:
            # 在樣本外數據上驗證
            print(f"\n最佳參數: {best_params}")
            print(f"樣本內分數: {best_score:.4f}")
            print("\n在樣本外數據上驗證...")
            
            test_result = self._evaluate_strategy(
                best_result['strategy'],
                self.test_data,
                return_full=True
            )
            
            if test_result.get('valid'):
                best_result['test_score'] = test_result.get('score', 0)
                best_result['test_result'] = test_result
                best_result['strategy_class'] = strategy_class
                
                print(f"樣本外分數: {test_result.get('score', 0):.4f}")
                print(f"樣本外總報酬: {test_result.get('total_return', 0)*100:.2f}%")
                print(f"樣本外夏普比率: {test_result.get('sharpe_ratio', 0):.2f}")
            else:
                print(f"樣本外驗證失敗: {test_result.get('reason', 'Unknown')}")
        
        return best_result
    
    def find_best_strategy(
        self,
        strategies_config: List[Dict],
        max_combinations_per_strategy: int = 50
    ) -> Dict:
        """
        自動測試多種策略，找到最佳策略
        
        Parameters:
        -----------
        strategies_config : list
            策略配置列表，每個元素包含：
            {
                'class': strategy_class,
                'param_grid': {...} 或 'param_ranges': {...},
                'method': 'grid_search' 或 'random_search'
            }
        max_combinations_per_strategy : int
            每個策略最大測試組合數
        
        Returns:
        --------
        dict
            最佳策略資訊
        """
        print("\n" + "="*60)
        print("開始自動化策略優化")
        print("="*60)
        print(f"訓練集: {self.train_data.index[0]} 至 {self.train_data.index[-1]} ({len(self.train_data)} 天)")
        print(f"測試集: {self.test_data.index[0]} 至 {self.test_data.index[-1]} ({len(self.test_data)} 天)")
        print(f"優化目標: {self.optimization_metric}")
        print("="*60)
        
        all_results = []
        
        for i, config in enumerate(strategies_config, 1):
            print(f"\n[{i}/{len(strategies_config)}] 測試策略: {config['class'].__name__}")
            
            result = self.optimize_strategy(
                strategy_class=config['class'],
                param_grid=config.get('param_grid'),
                param_ranges=config.get('param_ranges'),
                method=config.get('method', 'grid_search'),
                max_combinations=max_combinations_per_strategy
            )
            
            if result:
                result['strategy_name'] = config['class'].__name__
                all_results.append(result)
        
        if not all_results:
            print("\n警告: 沒有找到有效的策略")
            return None
        
        # 根據樣本外表現排序
        valid_results = [r for r in all_results if r.get('test_result') and r.get('test_result', {}).get('valid')]
        
        if not valid_results:
            print("\n警告: 沒有策略在樣本外數據上表現良好")
            # 使用樣本內結果
            valid_results = all_results
        
        # 排序：優先考慮樣本外表現，如果沒有則使用樣本內表現
        valid_results.sort(
            key=lambda x: x.get('test_score', x.get('score', -np.inf)),
            reverse=True
        )
        
        self.best_strategy_info = valid_results[0]
        
        # 打印結果摘要
        print("\n" + "="*60)
        print("優化結果摘要")
        print("="*60)
        
        for i, result in enumerate(valid_results[:5], 1):
            strategy_name = result.get('strategy_name', 'Unknown')
            params = result.get('params', {})
            train_score = result.get('score', 0)
            test_score = result.get('test_score', 0)
            test_return = result.get('test_result', {}).get('total_return', 0) * 100 if result.get('test_result') else 0
            
            print(f"\n{i}. {strategy_name}")
            print(f"   參數: {params}")
            print(f"   樣本內分數: {train_score:.4f}")
            if result.get('test_result'):
                print(f"   樣本外分數: {test_score:.4f}")
                print(f"   樣本外報酬: {test_return:.2f}%")
        
        print("\n" + "="*60)
        print(f"最佳策略: {self.best_strategy_info.get('strategy_name')}")
        print(f"最佳參數: {self.best_strategy_info.get('params')}")
        print("="*60)
        
        return self.best_strategy_info
    
    def get_best_strategy(self) -> Optional[BaseStrategy]:
        """獲取最佳策略實例"""
        if self.best_strategy_info:
            return self.best_strategy_info.get('strategy')
        return None
    
    def evaluate_on_full_data(self) -> Dict:
        """在完整數據上評估最佳策略"""
        if not self.best_strategy_info:
            print("尚未找到最佳策略")
            return None
        
        print("\n在完整數據上評估最佳策略...")
        strategy = self.best_strategy_info['strategy']
        
        result = self._evaluate_strategy(strategy, self.data, return_full=True)
        
        if result.get('valid'):
            print(f"完整數據總報酬: {result.get('total_return', 0)*100:.2f}%")
            print(f"完整數據夏普比率: {result.get('sharpe_ratio', 0):.2f}")
            print(f"完整數據最大回撤: {result.get('max_drawdown', 0)*100:.2f}%")
        
        return result

