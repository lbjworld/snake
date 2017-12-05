# coding: utf-8
import logging
import numpy as np
import pandas as pd

from data_loader import data_loader

logger = logging.getLogger(__name__)


class FastTradingEnv(object):

    VOLUME_SCALE_FACTOR = 1000000.0

    def __init__(self, name, days, use_adjust_close=True):
        self.name = name
        self.days = days

        data_df = data_loader(self.name)
        if data_df.empty:
            raise Exception('load stock[{name}] data error'.format(name=self.name))
        logger.debug('stock[{name}] data loaded'.format(name=self.name))

        close_column = 'Adj Close' if use_adjust_close else 'Close'
        data_df = data_df[['Open', 'High', 'Low', close_column, 'Volume']]
        data_df.columns = ['open', 'high', 'low', 'close', 'volume']
        data_df = data_df[~np.isnan(data_df.volume)]  # 跳过所有停牌日
        self.pct_change = data_df.pct_change().fillna(0.0) + 1.0  # 计算变化量
        data_df.volume = data_df.volume / FastTradingEnv.VOLUME_SCALE_FACTOR
        self.data = data_df
        self.reset()

    def reset(self):
        # we want continuous data
        high = len(self.data.index) - self.days
        if high <= 1:
            raise Exception('stock[{name}] data too short'.format(name=self.name))
        self.idx = np.random.randint(low=1, high=high)
        self.step = 0

    def step(self):
        current_idx = self.idx + self.step
        visible_data = np.zeros((self.days, len(self.data.columns)))
        if self.step > 0:
            current_data = self.data[self.idx:current_idx].as_matrix()
            visible_data[:current_data.shape[0]] += current_data
        obs = visible_data
        pct_change = 
        obs = {
            'pct_change': self.pct_change.iloc[current_idx].as_matrix(),
            'history': history,
        }
        self.step += 1
        done = self.step > self.days
        return obs, done
