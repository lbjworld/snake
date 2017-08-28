# coding: utf-8

from utils import run_backtest

if __name__ == '__main__':
    run_backtest(
        symbol='000333.SZ',
        # strategy='random_forecast.RandomForecastingStrategy',
        strategy='ma_cross.MovingAverageCrossStrategy',
        portfolio='utils.MarketOnClosePortfolio'
    )
