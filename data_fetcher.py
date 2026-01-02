"""
台灣股票數據獲取模組
支援從 yfinance 獲取台灣股票數據（使用 .TW 後綴）
"""
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')


class TWStockDataFetcher:
    """台灣股票數據獲取器"""
    
    def __init__(self):
        self.cache = {}
    
    def _format_ticker(self, symbol: str) -> str:
        """
        格式化股票代碼為 yfinance 格式
        例如: 2330 -> 2330.TW
        """
        if not symbol.endswith('.TW') and not symbol.endswith('.TWO'):
            return f"{symbol}.TW"
        return symbol
    
    def fetch_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        獲取股票歷史數據
        
        Parameters:
        -----------
        symbol : str
            股票代碼（例如: '2330' 或 '2330.TW'）
        start_date : str
            開始日期 (YYYY-MM-DD)
        end_date : str
            結束日期 (YYYY-MM-DD)
        interval : str
            數據間隔 ('1d', '1wk', '1mo')
        
        Returns:
        --------
        pd.DataFrame
            包含 OHLCV 數據的 DataFrame
        """
        ticker = self._format_ticker(symbol)
        cache_key = f"{ticker}_{start_date}_{end_date}_{interval}"
        
        if cache_key in self.cache:
            return self.cache[cache_key].copy()
        
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                raise ValueError(f"無法獲取 {symbol} 的數據，請檢查股票代碼和日期範圍")
            
            # 標準化欄位名稱
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            # 確保有必要的欄位
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"數據缺少必要欄位: {required_cols}")
            
            # 移除時區資訊
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # 排序索引
            df = df.sort_index()
            
            # 快取數據
            self.cache[cache_key] = df.copy()
            
            return df
            
        except Exception as e:
            raise Exception(f"獲取 {symbol} 數據時發生錯誤: {str(e)}")
    
    def fetch_multiple(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = '1d'
    ) -> dict:
        """
        獲取多檔股票的數據
        
        Returns:
        --------
        dict
            以股票代碼為 key 的字典，值為 DataFrame
        """
        data = {}
        for symbol in symbols:
            try:
                data[symbol] = self.fetch_data(symbol, start_date, end_date, interval)
            except Exception as e:
                print(f"警告: 無法獲取 {symbol} 的數據: {str(e)}")
                continue
        return data
    
    def clear_cache(self):
        """清除快取"""
        self.cache.clear()


def get_tw_stock_list() -> List[str]:
    """
    返回常見的台灣股票代碼列表
    可以擴展為從 API 或檔案讀取完整列表
    """
    return [
        '2330',  # 台積電
        '2317',  # 鴻海
        '2454',  # 聯發科
        '2308',  # 台達電
        '2303',  # 聯電
        '2412',  # 中華電
        '1301',  # 台塑
        '1303',  # 南亞
        '2891',  # 中信金
        '2882',  # 國泰金
    ]

