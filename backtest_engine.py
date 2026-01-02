"""
回測引擎核心模組
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
from strategy import BaseStrategy, Signal
from data_fetcher import TWStockDataFetcher


class Position:
    """持倉資訊"""
    
    def __init__(self):
        self.shares = 0
        self.avg_price = 0.0
        self.total_cost = 0.0
    
    def buy(self, shares: int, price: float):
        """買入股票"""
        if shares <= 0:
            return
        
        total_value = shares * price
        self.total_cost += total_value
        self.shares += shares
        self.avg_price = self.total_cost / self.shares if self.shares > 0 else 0
    
    def sell(self, shares: int, price: float) -> float:
        """賣出股票，返回收益"""
        if shares <= 0 or shares > self.shares:
            return 0.0
        
        proceeds = shares * price
        cost = shares * self.avg_price
        profit = proceeds - cost
        
        self.shares -= shares
        self.total_cost -= cost
        
        if self.shares == 0:
            self.avg_price = 0.0
            self.total_cost = 0.0
        
        return profit
    
    def get_value(self, current_price: float) -> float:
        """獲取當前持倉市值"""
        return self.shares * current_price
    
    def get_unrealized_profit(self, current_price: float) -> float:
        """獲取未實現損益"""
        if self.shares == 0:
            return 0.0
        return (current_price - self.avg_price) * self.shares


class BacktestEngine:
    """回測引擎"""
    
    def __init__(
        self,
        initial_capital: float = 1000000,
        commission: float = 0.001425,  # 台灣股票手續費率（0.1425%）
        tax: float = 0.003,  # 台灣股票交易稅（賣出時 0.3%）
        slippage: float = 0.001  # 滑點（0.1%）
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.tax = tax
        self.slippage = slippage
        
        self.position = Position()
        self.cash = initial_capital
        self.equity = []
        self.trades = []
        self.daily_returns = []
        
    def _calculate_cost(self, shares: int, price: float, is_buy: bool) -> float:
        """計算交易成本"""
        value = shares * price
        commission_cost = value * self.commission
        tax_cost = value * self.tax if not is_buy else 0
        slippage_cost = value * self.slippage
        return commission_cost + tax_cost + slippage_cost
    
    def execute_trade(self, signal: int, price: float, date: pd.Timestamp):
        """執行交易"""
        if signal == Signal.HOLD.value:
            return
        
        # 計算可用資金可買入的股數（以張為單位，1張=1000股）
        available_shares = int(self.cash / (price * 1000)) * 1000 if signal == Signal.BUY.value else self.position.shares
        
        if available_shares <= 0:
            return
        
        # 應用滑點
        execution_price = price * (1 + self.slippage) if signal == Signal.BUY.value else price * (1 - self.slippage)
        
        if signal == Signal.BUY.value:
            # 買入
            cost = available_shares * execution_price
            total_cost = cost + self._calculate_cost(available_shares, execution_price, True)
            
            if total_cost <= self.cash:
                self.position.buy(available_shares, execution_price)
                self.cash -= total_cost
                
                self.trades.append({
                    'date': date,
                    'action': 'BUY',
                    'shares': available_shares,
                    'price': execution_price,
                    'cost': total_cost
                })
        
        elif signal == Signal.SELL.value:
            # 賣出
            if self.position.shares > 0:
                shares_to_sell = min(available_shares, self.position.shares)
                proceeds = shares_to_sell * execution_price
                total_cost = self._calculate_cost(shares_to_sell, execution_price, False)
                net_proceeds = proceeds - total_cost
                
                profit = self.position.sell(shares_to_sell, execution_price)
                self.cash += net_proceeds
                
                self.trades.append({
                    'date': date,
                    'action': 'SELL',
                    'shares': shares_to_sell,
                    'price': execution_price,
                    'proceeds': net_proceeds,
                    'profit': profit
                })
    
    def update_equity(self, current_price: float, date: pd.Timestamp):
        """更新權益"""
        position_value = self.position.get_value(current_price)
        total_equity = self.cash + position_value
        
        self.equity.append({
            'date': date,
            'cash': self.cash,
            'position_value': position_value,
            'total_equity': total_equity,
            'shares': self.position.shares,
            'avg_price': self.position.avg_price
        })
    
    def run(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        執行回測
        
        Parameters:
        -----------
        data : pd.DataFrame
            股票數據
        strategy : BaseStrategy
            交易策略
        start_date : str, optional
            回測開始日期
        end_date : str, optional
            回測結束日期
        
        Returns:
        --------
        pd.DataFrame
            回測結果
        """
        # 重置狀態
        self.position = Position()
        self.cash = self.initial_capital
        self.equity = []
        self.trades = []
        self.daily_returns = []
        
        # 篩選日期範圍
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        if data.empty:
            raise ValueError("回測數據為空，請檢查日期範圍")
        
        # 生成交易信號
        signals = strategy.generate_signals(data)
        
        # 執行回測
        for date, row in data.iterrows():
            current_price = row['close']
            signal = signals.get(date, Signal.HOLD.value)
            
            # 執行交易
            self.execute_trade(signal, current_price, date)
            
            # 更新權益
            self.update_equity(current_price, date)
        
        # 轉換為 DataFrame
        equity_df = pd.DataFrame(self.equity)
        equity_df.set_index('date', inplace=True)
        
        # 計算每日報酬率
        equity_df['returns'] = equity_df['total_equity'].pct_change()
        equity_df['cumulative_returns'] = (1 + equity_df['returns']).cumprod() - 1
        
        return equity_df
    
    def get_trades(self) -> pd.DataFrame:
        """獲取交易記錄"""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)
    
    def get_final_stats(self) -> Dict:
        """獲取最終統計資訊"""
        if not self.equity:
            return {}
        
        equity_df = pd.DataFrame(self.equity)
        equity_df.set_index('date', inplace=True)
        
        final_equity = equity_df['total_equity'].iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        returns = equity_df['total_equity'].pct_change().dropna()
        
        if len(returns) > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
            max_drawdown = self._calculate_max_drawdown(equity_df['total_equity'])
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'win_rate': self._calculate_win_rate()
        }
    
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """計算最大回撤"""
        cumulative = (1 + equity.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_win_rate(self) -> float:
        """計算勝率"""
        if not self.trades:
            return 0.0
        
        trades_df = pd.DataFrame(self.trades)
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        
        if len(sell_trades) == 0:
            return 0.0
        
        winning_trades = len(sell_trades[sell_trades['profit'] > 0])
        return winning_trades / len(sell_trades)

