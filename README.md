# 台灣股票回測系統

一個功能完整的台灣股票回測系統，支援多種技術分析策略和詳細的績效分析。

## 功能特色

- 📊 **數據獲取**: 支援從 yfinance 獲取台灣股票數據（使用 .TW 後綴）
- 📈 **多種策略**: 內建移動平均、RSI、MACD 等策略，並支援自訂策略
- 🔄 **回測引擎**: 完整的回測引擎，考慮手續費、交易稅和滑點
- 📉 **績效分析**: 詳細的績效指標和視覺化圖表
- 🎯 **策略比較**: 支援同時比較多個策略的表現
- 🔍 **參數優化**: 自動尋找最優策略參數，支援網格搜索、隨機搜索和貝葉斯優化
- 🤖 **自動化優化**: 自動測試多種策略，包含樣本內/樣本外測試，避免過度擬合

## 安裝

1. 安裝依賴套件：

```bash
pip install -r requirements.txt
```

2. 如果使用 TA-Lib，可能需要額外安裝：

```bash
# macOS
brew install ta-lib
pip install TA-Lib

# Linux
# 請參考 TA-Lib 官方文檔
```

## 快速開始

### 基本使用

```python
from data_fetcher import TWStockDataFetcher
from strategy import MovingAverageStrategy
from backtest_engine import BacktestEngine
from performance import PerformanceAnalyzer

# 獲取數據
fetcher = TWStockDataFetcher()
data = fetcher.fetch_data("2330", "2022-01-01", "2023-12-31")

# 建立策略
strategy = MovingAverageStrategy(short_window=5, long_window=20)

# 執行回測
engine = BacktestEngine(initial_capital=1000000)
equity_df = engine.run(data, strategy)

# 分析績效
analyzer = PerformanceAnalyzer(equity_df, 1000000)
print(analyzer.generate_report())
analyzer.plot_equity_curve()
```

### 自動化策略優化（推薦）

自動測試多種策略並找到最佳策略，包含樣本內/樣本外測試以避免過度擬合：

```bash
# 使用預設參數（推薦）
python auto_run.py

# 自訂參數
python auto_run.py --symbol 2330 --start 2020-01-01 --end 2023-12-31 --metric custom

# 查看所有選項
python auto_run.py --help
```

或使用主程式：

```bash
# 使用預設參數
python main.py

# 指定股票代碼
python main.py --auto 2330

# 完整參數
python main.py --auto 2330 2020-01-01 2023-12-31
```

## 專案結構

```
backtesting/
├── data_fetcher.py      # 數據獲取模組
├── strategy.py          # 策略模組
├── backtest_engine.py   # 回測引擎
├── performance.py       # 績效分析模組
├── optimizer.py         # 參數優化模組
├── auto_optimizer.py    # 自動化優化模組（樣本內/樣本外測試）
├── main.py              # 主程式（提供基本回測功能）
├── auto_run.py          # 自動化優化執行腳本（推薦使用）
├── requirements.txt     # 依賴套件
└── README.md           # 說明文檔
```

## 內建策略

### 1. 移動平均策略 (MovingAverageStrategy)

使用短期和長期移動平均線的交叉來產生交易信號。

```python
strategy = MovingAverageStrategy(short_window=5, long_window=20)
```

### 2. RSI 策略 (RSIStrategy)

使用相對強弱指標（RSI）來判斷超買超賣。

```python
strategy = RSIStrategy(period=14, oversold=30, overbought=70)
```

### 3. MACD 策略 (MACDStrategy)

使用 MACD 指標的交叉來產生交易信號。

```python
strategy = MACDStrategy(fast=12, slow=26, signal=9)
```

### 4. 組合策略 (CombinedStrategy)

結合移動平均和 RSI 策略。

```python
strategy = CombinedStrategy(ma_short=5, ma_long=20, rsi_period=14)
```

## 自訂策略

您可以繼承 `BaseStrategy` 類別來建立自己的策略：

```python
from strategy import BaseStrategy, Signal
import pandas as pd

class MyCustomStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="MyStrategy")
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        # 計算您的技術指標
        df = data.copy()
        # ... 您的指標計算 ...
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # 生成交易信號
        df = self.calculate_indicators(data)
        signals = pd.Series(0, index=df.index)
        # ... 您的信號邏輯 ...
        return signals
```

## 回測參數

`BacktestEngine` 支援以下參數：

- `initial_capital`: 初始資金（預設: 1,000,000）
- `commission`: 手續費率（預設: 0.001425，即 0.1425%）
- `tax`: 交易稅率，僅賣出時收取（預設: 0.003，即 0.3%）
- `slippage`: 滑點（預設: 0.001，即 0.1%）

## 績效指標

系統會計算以下績效指標：

- **總報酬率**: 總投資報酬率
- **年化報酬率**: 年化後的報酬率
- **波動率**: 報酬率的標準差（年化）
- **夏普比率**: 風險調整後的報酬率
- **最大回撤**: 從峰值到谷值的最大跌幅
- **卡爾瑪比率**: 年化報酬率與最大回撤的比值

## 視覺化圖表

- **權益曲線**: 顯示策略權益隨時間的變化
- **回撤分析**: 顯示回撤情況
- **月度報酬率熱力圖**: 顯示各月份的報酬率
- **報酬率分布**: 顯示報酬率的統計分布

## 台灣股票代碼格式

系統會自動將股票代碼轉換為 yfinance 格式：
- `2330` → `2330.TW`（上市股票）
- `2330.TWO` → `2330.TWO`（上櫃股票）

## 自動化策略優化（推薦）

系統提供完整的自動化優化功能，可以自動測試多種策略並找到最佳策略，**包含樣本內/樣本外測試以避免過度擬合**。

### 快速開始

最簡單的方式是直接運行自動優化腳本：

```bash
python auto_run.py --symbol 2330
```

系統會自動：
1. 獲取股票數據
2. 將數據分割為訓練集（70%）和測試集（30%）
3. 在訓練集上優化多種策略的參數
4. 在測試集上驗證策略表現
5. 在完整數據上評估最佳策略
6. 生成完整的績效報告和圖表

### 自動化優化流程

```python
from auto_optimizer import AutoStrategyOptimizer
from data_fetcher import TWStockDataFetcher

# 獲取數據
fetcher = TWStockDataFetcher()
data = fetcher.fetch_data("2330", "2020-01-01", "2023-12-31")

# 創建優化器（自動分割訓練/測試集）
optimizer = AutoStrategyOptimizer(
    data=data,
    initial_capital=1000000,
    train_ratio=0.7,  # 70% 用於訓練，30% 用於測試
    optimization_metric='custom'
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
    # ... 更多策略
]

# 自動尋找最佳策略
best_strategy_info = optimizer.find_best_strategy(strategies_config)

# 獲取最佳策略
best_strategy = optimizer.get_best_strategy()
```

### 避免過度擬合

系統使用以下方法避免過度擬合：

1. **樣本內/樣本外測試**：在訓練集上優化參數，在測試集上驗證表現
2. **最少交易次數要求**：過濾交易次數過少的策略
3. **樣本外驗證**：優先選擇在樣本外數據上表現良好的策略
4. **綜合評分**：考慮報酬率、風險調整報酬和勝率

## 手動參數優化

如果您想手動控制優化過程，系統也提供強大的參數優化功能，可以自動尋找最優策略參數。支援三種優化方法：

### 1. 網格搜索 (Grid Search)

遍歷所有參數組合，找到最佳參數：

```python
from optimizer import StrategyOptimizer
from strategy import MovingAverageStrategy

optimizer = StrategyOptimizer(
    strategy_class=MovingAverageStrategy,
    data=data,
    optimization_metric='sharpe_ratio'  # 優化目標
)

param_grid = {
    'short_window': [3, 5, 7, 10, 12],
    'long_window': [15, 20, 25, 30, 35]
}

result = optimizer.grid_search(param_grid)
optimizer.print_optimization_summary()
```

### 2. 隨機搜索 (Random Search)

在參數範圍內隨機搜索，適合參數空間較大的情況：

```python
param_ranges = {
    'period': (10, 20, 'int'),
    'oversold': (20, 40, 'float'),
    'overbought': (60, 80, 'float')
}

result = optimizer.random_search(param_ranges, n_iter=100)
```

### 3. scipy 優化 (Differential Evolution)

使用進化算法進行優化，效率較高：

```python
result = optimizer.scipy_optimize(
    param_ranges,
    method='differential_evolution'
)
```

### 優化目標指標

- `sharpe_ratio`: 夏普比率（預設）
- `total_return`: 總報酬率
- `calmar_ratio`: 卡爾瑪比率
- `custom`: 自訂評分函數（綜合考慮報酬率和風險）

### 查看優化結果

```python
# 獲取最佳參數
best_params = optimizer.best_params
best_score = optimizer.best_score

# 獲取前 N 名結果
top_results = optimizer.get_top_results(n=10)

# 使用最佳參數進行回測
best_strategy = MovingAverageStrategy(**best_params)
```

## 注意事項

1. **數據來源**: 本系統使用 yfinance 獲取數據，請確保網路連線正常
2. **交易成本**: 預設參數適用於台灣股票市場，可根據實際情況調整
3. **數據品質**: 請確認獲取的數據完整且正確
4. **策略參數**: 不同股票可能需要不同的策略參數，建議使用參數優化功能自動尋找最優參數
5. **過度擬合**: 參數優化可能導致過度擬合，建議使用樣本外數據驗證策略效果

## 範例股票代碼

- `2330`: 台積電
- `2317`: 鴻海
- `2454`: 聯發科
- `2308`: 台達電
- `2303`: 聯電
- `2412`: 中華電

## 授權

本專案僅供學習和研究使用。

## 貢獻

歡迎提交 Issue 和 Pull Request！

# Cursor-Backtest-System
