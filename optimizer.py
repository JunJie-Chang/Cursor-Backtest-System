"""
策略參數優化模組
支援網格搜索、隨機搜索和貝葉斯優化等方法來尋找最優策略參數
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from itertools import product
import random
from scipy.optimize import minimize, differential_evolution
from backtest_engine import BacktestEngine
from performance import PerformanceAnalyzer
from strategy import BaseStrategy
import warnings
warnings.filterwarnings('ignore')


class StrategyOptimizer:
    """策略參數優化器"""
    
    def __init__(
        self,
        strategy_class: type,
        data: pd.DataFrame,
        initial_capital: float = 1000000,
        optimization_metric: str = 'sharpe_ratio',
        engine_params: Optional[Dict] = None
    ):
        """
        初始化優化器
        
        Parameters:
        -----------
        strategy_class : type
            策略類別（如 MovingAverageStrategy）
        data : pd.DataFrame
            回測數據
        initial_capital : float
            初始資金
        optimization_metric : str
            優化目標指標 ('sharpe_ratio', 'total_return', 'calmar_ratio', 'custom')
        engine_params : dict, optional
            回測引擎參數
        """
        self.strategy_class = strategy_class
        self.data = data
        self.initial_capital = initial_capital
        self.optimization_metric = optimization_metric
        self.engine_params = engine_params or {}
        
        self.results = []
        self.best_params = None
        self.best_score = None
        self.best_result = None
    
    def _evaluate_params(
        self,
        params: Dict,
        return_full_stats: bool = False
    ) -> float:
        """
        評估參數組合的績效
        
        Parameters:
        -----------
        params : dict
            策略參數
        return_full_stats : bool
            是否返回完整統計資訊
        
        Returns:
        --------
        float or dict
            優化目標分數或完整統計資訊
        """
        try:
            # 創建策略實例
            strategy = self.strategy_class(**params)
            
            # 執行回測
            engine = BacktestEngine(
                initial_capital=self.initial_capital,
                **self.engine_params
            )
            equity_df = engine.run(self.data, strategy)
            
            # 計算績效指標
            analyzer = PerformanceAnalyzer(equity_df, self.initial_capital)
            stats = engine.get_final_stats()
            metrics = analyzer.calculate_metrics()
            
            # 合併統計資訊
            full_stats = {**stats, **metrics}
            
            # 計算優化目標分數
            if self.optimization_metric == 'sharpe_ratio':
                score = stats.get('sharpe_ratio', 0)
            elif self.optimization_metric == 'total_return':
                score = stats.get('total_return', 0)
            elif self.optimization_metric == 'calmar_ratio':
                try:
                    annual_return_str = metrics.get('年化報酬率', '0%')
                    annual_return = float(str(annual_return_str).replace('%', '').replace('$', '').strip()) / 100
                    max_drawdown_str = metrics.get('最大回撤', '0%')
                    max_drawdown = abs(float(str(max_drawdown_str).replace('%', '').replace('$', '').strip()) / 100)
                    score = annual_return / max_drawdown if max_drawdown > 0 else 0
                except (ValueError, AttributeError):
                    score = 0
            elif self.optimization_metric == 'custom':
                # 自訂評分：綜合考慮報酬率和風險
                total_return = stats.get('total_return', 0)
                sharpe = stats.get('sharpe_ratio', 0)
                try:
                    max_dd_str = metrics.get('最大回撤', '0%')
                    max_dd = abs(float(str(max_dd_str).replace('%', '').replace('$', '').strip()) / 100)
                    # 綜合評分：報酬率 * 夏普比率 / (1 + 最大回撤)
                    score = total_return * sharpe / (1 + max_dd) if max_dd > 0 else total_return * sharpe
                except (ValueError, AttributeError):
                    score = total_return * sharpe
            else:
                score = stats.get('sharpe_ratio', 0)
            
            if return_full_stats:
                return score, full_stats, equity_df, engine
            return score
            
        except Exception as e:
            # 如果參數組合無效，返回極小值
            if return_full_stats:
                return -np.inf, {}, None, None
            return -np.inf
    
    def grid_search(
        self,
        param_grid: Dict[str, List],
        verbose: bool = True
    ) -> Dict:
        """
        網格搜索優化
        
        Parameters:
        -----------
        param_grid : dict
            參數搜索空間，格式：{'param_name': [value1, value2, ...]}
        verbose : bool
            是否顯示進度
        
        Returns:
        --------
        dict
            最佳參數和結果
        """
        if verbose:
            print("開始網格搜索優化...")
            print(f"參數搜索空間: {param_grid}")
        
        # 生成所有參數組合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        total_combinations = np.prod([len(v) for v in param_values])
        
        if verbose:
            print(f"總共需要測試 {total_combinations} 個參數組合")
        
        self.results = []
        best_score = -np.inf
        best_params = None
        best_result = None
        
        # 遍歷所有參數組合
        for i, combination in enumerate(product(*param_values), 1):
            params = dict(zip(param_names, combination))
            
            try:
                score, stats, equity_df, engine = self._evaluate_params(params, return_full_stats=True)
                
                result = {
                    'params': params,
                    'score': score,
                    'stats': stats,
                    'equity_df': equity_df,
                    'engine': engine
                }
                self.results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_result = result
                
                if verbose and i % max(1, total_combinations // 10) == 0:
                    print(f"進度: {i}/{total_combinations} ({i/total_combinations*100:.1f}%) - "
                          f"當前最佳分數: {best_score:.4f}")
            
            except Exception as e:
                if verbose:
                    print(f"參數組合 {params} 執行失敗: {str(e)}")
                continue
        
        self.best_params = best_params
        self.best_score = best_score
        self.best_result = best_result
        
        if verbose:
            print(f"\n網格搜索完成！")
            print(f"最佳參數: {best_params}")
            print(f"最佳分數 ({self.optimization_metric}): {best_score:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_result': best_result,
            'all_results': self.results
        }
    
    def random_search(
        self,
        param_ranges: Dict[str, Tuple],
        n_iter: int = 100,
        verbose: bool = True
    ) -> Dict:
        """
        隨機搜索優化
        
        Parameters:
        -----------
        param_ranges : dict
            參數範圍，格式：{'param_name': (min, max, type)}
            type 可以是 'int' 或 'float'
        n_iter : int
            隨機搜索次數
        verbose : bool
            是否顯示進度
        
        Returns:
        --------
        dict
            最佳參數和結果
        """
        if verbose:
            print("開始隨機搜索優化...")
            print(f"參數範圍: {param_ranges}")
            print(f"搜索次數: {n_iter}")
        
        self.results = []
        best_score = -np.inf
        best_params = None
        best_result = None
        
        for i in range(n_iter):
            # 隨機生成參數
            params = {}
            for param_name, (min_val, max_val, param_type) in param_ranges.items():
                if param_type == 'int':
                    params[param_name] = random.randint(int(min_val), int(max_val))
                else:  # float
                    params[param_name] = random.uniform(min_val, max_val)
            
            try:
                score, stats, equity_df, engine = self._evaluate_params(params, return_full_stats=True)
                
                result = {
                    'params': params,
                    'score': score,
                    'stats': stats,
                    'equity_df': equity_df,
                    'engine': engine
                }
                self.results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_result = result
                
                if verbose and (i + 1) % max(1, n_iter // 10) == 0:
                    print(f"進度: {i+1}/{n_iter} ({(i+1)/n_iter*100:.1f}%) - "
                          f"當前最佳分數: {best_score:.4f}")
            
            except Exception as e:
                if verbose:
                    print(f"參數組合 {params} 執行失敗: {str(e)}")
                continue
        
        self.best_params = best_params
        self.best_score = best_score
        self.best_result = best_result
        
        if verbose:
            print(f"\n隨機搜索完成！")
            print(f"最佳參數: {best_params}")
            print(f"最佳分數 ({self.optimization_metric}): {best_score:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_result': best_result,
            'all_results': self.results
        }
    
    def _objective_function(self, x: np.ndarray, param_info: List[Tuple]) -> float:
        """
        目標函數（用於 scipy.optimize）
        
        Parameters:
        -----------
        x : np.ndarray
            參數值陣列
        param_info : list
            參數資訊列表，每個元素為 (name, min_val, max_val, type)
        
        Returns:
        --------
        float
            負分數（因為 minimize 會最小化目標函數）
        """
        params = {}
        for i, (name, min_val, max_val, param_type) in enumerate(param_info):
            if param_type == 'int':
                params[name] = int(np.clip(x[i], min_val, max_val))
            else:
                params[name] = float(np.clip(x[i], min_val, max_val))
        
        score = self._evaluate_params(params)
        return -score  # 返回負值，因為 minimize 會最小化
    
    def scipy_optimize(
        self,
        param_ranges: Dict[str, Tuple],
        method: str = 'differential_evolution',
        verbose: bool = True
    ) -> Dict:
        """
        使用 scipy.optimize 進行優化
        
        Parameters:
        -----------
        param_ranges : dict
            參數範圍，格式：{'param_name': (min, max, type)}
        method : str
            優化方法 ('differential_evolution', 'minimize')
        verbose : bool
            是否顯示進度
        
        Returns:
        --------
        dict
            最佳參數和結果
        """
        if verbose:
            print(f"開始使用 {method} 優化...")
            print(f"參數範圍: {param_ranges}")
        
        # 準備參數資訊
        param_names = list(param_ranges.keys())
        param_info = [(name, *param_ranges[name]) for name in param_names]
        
        # 準備邊界
        bounds = [(min_val, max_val) for _, min_val, max_val, _ in param_info]
        
        # 初始值（使用範圍中點）
        x0 = []
        for _, min_val, max_val, param_type in param_info:
            if param_type == 'int':
                x0.append(int((min_val + max_val) / 2))
            else:
                x0.append((min_val + max_val) / 2)
        
        if method == 'differential_evolution':
            result = differential_evolution(
                self._objective_function,
                bounds,
                args=(param_info,),
                seed=42,
                maxiter=50,
                popsize=15,
                polish=True
            )
        else:
            result = minimize(
                self._objective_function,
                x0,
                args=(param_info,),
                method='L-BFGS-B',
                bounds=bounds
            )
        
        # 轉換結果為參數字典
        best_params = {}
        for i, (name, min_val, max_val, param_type) in enumerate(param_info):
            if param_type == 'int':
                best_params[name] = int(np.clip(result.x[i], min_val, max_val))
            else:
                best_params[name] = float(np.clip(result.x[i], min_val, max_val))
        
        # 評估最佳參數
        best_score, stats, equity_df, engine = self._evaluate_params(best_params, return_full_stats=True)
        
        self.best_params = best_params
        self.best_score = best_score
        self.best_result = {
            'params': best_params,
            'score': best_score,
            'stats': stats,
            'equity_df': equity_df,
            'engine': engine
        }
        
        if verbose:
            print(f"\n優化完成！")
            print(f"最佳參數: {best_params}")
            print(f"最佳分數 ({self.optimization_metric}): {best_score:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_result': self.best_result
        }
    
    def get_top_results(self, n: int = 10) -> List[Dict]:
        """
        獲取前 N 個最佳結果
        
        Parameters:
        -----------
        n : int
            返回的結果數量
        
        Returns:
        --------
        list
            排序後的結果列表
        """
        if not self.results:
            return []
        
        sorted_results = sorted(self.results, key=lambda x: x['score'], reverse=True)
        return sorted_results[:n]
    
    def print_optimization_summary(self):
        """打印優化摘要"""
        if not self.best_result:
            print("尚未執行優化")
            return
        
        print("\n" + "="*60)
        print("優化結果摘要")
        print("="*60)
        print(f"優化目標: {self.optimization_metric}")
        print(f"最佳參數: {self.best_params}")
        print(f"最佳分數: {self.best_score:.4f}")
        
        if self.best_result.get('stats'):
            stats = self.best_result['stats']
            print("\n績效指標:")
            print(f"  總報酬率: {stats.get('total_return_pct', 'N/A'):.2f}%")
            print(f"  夏普比率: {stats.get('sharpe_ratio', 'N/A'):.2f}")
            print(f"  最大回撤: {stats.get('max_drawdown', 'N/A')*100:.2f}%")
            print(f"  勝率: {stats.get('win_rate', 'N/A')*100:.2f}%")
            print(f"  總交易次數: {stats.get('total_trades', 'N/A')}")
        
        if len(self.results) > 1:
            print(f"\n總共測試了 {len(self.results)} 個參數組合")
            print(f"前 5 名結果:")
            top_results = self.get_top_results(5)
            for i, result in enumerate(top_results, 1):
                print(f"  {i}. 參數: {result['params']}, 分數: {result['score']:.4f}")
        
        print("="*60)

