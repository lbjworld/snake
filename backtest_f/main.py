# coding: utf-8

from datetime import datetime
import pandas_datareader as pdr

from f.portfolios import MarketOnClosePortfolio
from f.strategy.ma_cross import MovingAverageCrossStrategy
# from f.strategy.random_forecast import RandomForecastingStrategy


def run_backtest(symbol, date_range=(datetime(2016, 8, 29), datetime.now())):
    # get data from yahoo
    bars = pdr.get_data_yahoo(symbol, start=date_range[0], end=date_range[1])
    print 'stock bars: ', bars.head(10)
    # create strategy class and get signals
    strategy_inst = MovingAverageCrossStrategy(symbol, bars)
    signals = strategy_inst.generate_signals()
    print 'signals', signals.head()
    # create a portfolio
    portfolio_inst = MarketOnClosePortfolio(
        symbol, bars, signals, initial_capital=100000.0, shares_per_position=1000
    )
    returns = portfolio_inst.backtest_portfolio()

    print 'head returns:', returns.head(10)
    print 'tail returns:', returns.tail(10)
    return returns


if __name__ == '__main__':
    run_backtest(
        # symbol='000333.SZ',
        # symbol='000034.SZ',
        symbol='600016.SH',
    )
