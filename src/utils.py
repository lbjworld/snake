# coding: utf-8
import importlib
from datetime import datetime
import pandas as pd
import pandas_datareader as pdr

from backtest import Portfolio


class MarketOnTickerPortfolio(Portfolio):
    """Inherits Portfolio to create a system that purchases 100 units of
    a particular symbol upon a long/short signal, assuming the market
    open price of a bar.

    In addition, there are zero transaction costs and cash can be immediately
    borrowed for shorting (no margin posting or interest requirements).

    Requires:
    symbol - A stock symbol which forms the basis of the portfolio.
    bars - A DataFrame of bars for a symbol set.
    signals - A pandas DataFrame of signals (1, 0, -1) for each symbol.
    initial_capital - The amount in cash at the start of the portfolio."""

    TICKER_KEY = None

    def __init__(self, symbol, bars, signals, initial_capital=100000.0):
        self.symbol = symbol
        self.bars = bars
        self.signals = signals
        self.initial_capital = float(initial_capital)
        self.positions = self.generate_positions()

    def generate_positions(self):
        """Creates a 'positions' DataFrame that simply longs or shorts
        100 of the particular symbol based on the forecast signals of
        {1, 0, -1} from the signals DataFrame."""
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        positions[self.symbol] = 1000 * self.signals['signal']
        return positions

    def backtest_portfolio(self):
        """Constructs a portfolio from the positions DataFrame by
        assuming the ability to trade at the precise market open price
        of each bar (an unrealistic assumption!).

        Calculates the total of cash and the holdings (market price of
        each position per bar), in order to generate an equity curve
        ('total') and a set of bar-based returns ('returns').

        Returns the portfolio object to be used elsewhere."""

        # Construct the portfolio DataFrame to use the same index
        # as 'positions' and with a set of 'trading orders' in the
        # 'pos_diff' object, assuming market open prices.
        portfolio = pd.DataFrame(index=self.positions.index).fillna(0.0)
        pos_diff = self.positions.diff()

        # Create the 'holdings' and 'cash' series by running through
        # the trades and adding/subtracting the relevant quantity from
        # each column
        portfolio['holdings'] = (self.positions.multiply(self.bars[self.TICKER_KEY], axis='index')).sum(axis=1)
        portfolio['cash'] = self.initial_capital - (pos_diff.multiply(self.bars[self.TICKER_KEY], axis='index')).sum(axis=1).cumsum(axis=0)

        # Finalise the total and bar-based returns based on the 'cash'
        # and 'holdings' figures for the portfolio
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()
        return portfolio


class MarketOnOpenPortfolio(MarketOnTickerPortfolio):
    TICKER_KEY = 'Open'


class MarketOnClosePortfolio(MarketOnTickerPortfolio):
    TICKER_KEY = 'Adj Close'


def run_backtest(
        symbol, strategy, portfolio, date_range=(datetime(2015, 8, 29), datetime.now())):
    def import_class_by_name(path_name):
        rs = path_name.split('.')
        module_path, class_name = '.'.join(rs[:-1]), rs[-1]
        target_class = getattr(importlib.import_module(module_path), class_name)
        return target_class
    # get data from yahoo
    bars = pdr.get_data_yahoo(symbol, start=date_range[0], end=date_range[1])
    print 'stock bars: ', bars.head(10)
    # create strategy class and get signals
    strategy_class = import_class_by_name(strategy)
    strategy_inst = strategy_class(symbol, bars)
    signals = strategy_inst.generate_signals()
    # create a portfolio
    portfolio_class = import_class_by_name(portfolio)
    portfolio_inst = portfolio_class(symbol, bars, signals, initial_capital=100000.0)
    returns = portfolio_inst.backtest_portfolio()

    print 'head returns:', returns.head(10)
    print 'tail returns:', returns.tail(10)
    return returns
