"""
台灣股票回測系統
"""

__version__ = "1.0.0"

from .data_fetcher import TWStockDataFetcher
from .strategy import (
    BaseStrategy,
    MovingAverageStrategy,
    RSIStrategy,
    MACDStrategy,
    CombinedStrategy,
    Signal
)
from .backtest_engine import BacktestEngine, Position
from .performance import PerformanceAnalyzer

__all__ = [
    'TWStockDataFetcher',
    'BaseStrategy',
    'MovingAverageStrategy',
    'RSIStrategy',
    'MACDStrategy',
    'CombinedStrategy',
    'Signal',
    'BacktestEngine',
    'Position',
    'PerformanceAnalyzer'
]

