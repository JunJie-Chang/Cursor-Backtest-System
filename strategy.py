"""
策略基類和範例策略
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional
from enum import Enum


class Signal(Enum):
    """交易信號"""
    BUY = 1
    SELL = -1
    HOLD = 0


class BaseStrategy(ABC):
    """策略基類"""
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.indicators = {}
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        生成交易信號
        
        Parameters:
        -----------
        data : pd.DataFrame
            包含 OHLCV 數據的 DataFrame
        
        Returns:
        --------
        pd.Series
            交易信號序列 (1: 買入, -1: 賣出, 0: 持有)
        """
        pass
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        計算技術指標（子類可以覆寫）
        
        Parameters:
        -----------
        data : pd.DataFrame
            包含 OHLCV 數據的 DataFrame
        
        Returns:
        --------
        pd.DataFrame
            包含原始數據和指標的 DataFrame
        """
        return data.copy()
    
    def get_params(self) -> Dict:
        """返回策略參數"""
        return {}


class MovingAverageStrategy(BaseStrategy):
    """移動平均線策略"""
    
    def __init__(self, short_window: int = 5, long_window: int = 20):
        super().__init__(name=f"MA_{short_window}_{long_window}")
        self.short_window = short_window
        self.long_window = long_window
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """計算移動平均線"""
        df = data.copy()
        df['ma_short'] = df['close'].rolling(window=self.short_window).mean()
        df['ma_long'] = df['close'].rolling(window=self.long_window).mean()
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信號"""
        df = self.calculate_indicators(data)
        
        signals = pd.Series(0, index=df.index)
        
        # 黃金交叉：短期均線上穿長期均線 -> 買入
        # 死亡交叉：短期均線下穿長期均線 -> 賣出
        signals[df['ma_short'] > df['ma_long']] = Signal.BUY.value
        signals[df['ma_short'] < df['ma_long']] = Signal.SELL.value
        
        # 處理交叉點
        ma_diff = df['ma_short'] - df['ma_long']
        ma_diff_prev = ma_diff.shift(1)
        
        # 黃金交叉
        golden_cross = (ma_diff > 0) & (ma_diff_prev <= 0)
        signals[golden_cross] = Signal.BUY.value
        
        # 死亡交叉
        death_cross = (ma_diff < 0) & (ma_diff_prev >= 0)
        signals[death_cross] = Signal.SELL.value
        
        return signals
    
    def get_params(self) -> Dict:
        return {
            'short_window': self.short_window,
            'long_window': self.long_window
        }


class RSIStrategy(BaseStrategy):
    """RSI 策略"""
    
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__(name=f"RSI_{period}")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """計算 RSI 指標"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """計算 RSI 指標"""
        df = data.copy()
        df['rsi'] = self.calculate_rsi(df['close'], self.period)
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信號"""
        df = self.calculate_indicators(data)
        
        signals = pd.Series(0, index=df.index)
        
        # RSI < 超賣線 -> 買入
        signals[df['rsi'] < self.oversold] = Signal.BUY.value
        
        # RSI > 超買線 -> 賣出
        signals[df['rsi'] > self.overbought] = Signal.SELL.value
        
        return signals
    
    def get_params(self) -> Dict:
        return {
            'period': self.period,
            'oversold': self.oversold,
            'overbought': self.overbought
        }


class MACDStrategy(BaseStrategy):
    """MACD 策略"""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(name=f"MACD_{fast}_{slow}_{signal}")
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def calculate_macd(self, prices: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
        """計算 MACD 指標"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return pd.DataFrame({
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        })
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """計算 MACD 指標"""
        df = data.copy()
        macd_data = self.calculate_macd(df['close'], self.fast, self.slow, self.signal)
        df = pd.concat([df, macd_data], axis=1)
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信號"""
        df = self.calculate_indicators(data)
        
        signals = pd.Series(0, index=df.index)
        
        # MACD 上穿信號線 -> 買入
        # MACD 下穿信號線 -> 賣出
        macd_diff = df['macd'] - df['signal']
        macd_diff_prev = macd_diff.shift(1)
        
        # 黃金交叉
        golden_cross = (macd_diff > 0) & (macd_diff_prev <= 0)
        signals[golden_cross] = Signal.BUY.value
        
        # 死亡交叉
        death_cross = (macd_diff < 0) & (macd_diff_prev >= 0)
        signals[death_cross] = Signal.SELL.value
        
        return signals
    
    def get_params(self) -> Dict:
        return {
            'fast': self.fast,
            'slow': self.slow,
            'signal': self.signal
        }


class CombinedStrategy(BaseStrategy):
    """組合策略（移動平均 + RSI）"""
    
    def __init__(self, ma_short: int = 5, ma_long: int = 20, rsi_period: int = 14):
        super().__init__(name="Combined_MA_RSI")
        self.ma_strategy = MovingAverageStrategy(ma_short, ma_long)
        self.rsi_strategy = RSIStrategy(rsi_period)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """計算所有指標"""
        df = self.ma_strategy.calculate_indicators(data)
        df = self.rsi_strategy.calculate_indicators(df)
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成組合信號"""
        ma_signals = self.ma_strategy.generate_signals(data)
        rsi_signals = self.rsi_strategy.generate_signals(data)
        
        # 結合信號：需要兩個策略都同意才交易
        signals = pd.Series(0, index=data.index)
        
        # 買入：MA 看多 且 RSI 不超買
        buy_condition = (ma_signals == Signal.BUY.value) & (rsi_signals != Signal.SELL.value)
        signals[buy_condition] = Signal.BUY.value
        
        # 賣出：MA 看空 或 RSI 超買
        sell_condition = (ma_signals == Signal.SELL.value) | (rsi_signals == Signal.SELL.value)
        signals[sell_condition] = Signal.SELL.value
        
        return signals
    
    def get_params(self) -> Dict:
        return {
            **self.ma_strategy.get_params(),
            **self.rsi_strategy.get_params()
        }

