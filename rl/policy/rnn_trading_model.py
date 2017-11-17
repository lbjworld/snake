# coding: utf-8
from __future__ import unicode_literals

from resnet import ResnetBuilder
from base_trading_model import BaseTradingModel


class RNNTradingModel(BaseTradingModel):
    def __init__(self):
        # input: (channel, row, col) -> (1, episode_days, ticker)
        # output: action probability
        self._model = ResnetBuilder.build_resnet_18((1, 200, 5), 2)
        ResnetBuilder.check_model(self._model)
