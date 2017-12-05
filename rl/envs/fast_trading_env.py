# coding: utf-8
import logging
import numpy as np

from data_loader import data_loader

logger = logging.getLogger(__name__)


class FastTradingEnv(object):

    VOLUME_SCALE_FACTOR = 1000000.0

    def __init__(self, name, days, use_adjust_close=True, trading_cost_bps=1e-3):
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

        self.trading_cost_pct_change = 1.0 - trading_cost_bps
        self.actions = np.zeros(self.steps)
        self.navs = np.ones(self.steps)

        self.reset()

    @property
    def action_options(self):
        return [0, 1]

    def reset(self):
        # we want continuous data
        high = len(self.data.index) - self.days
        if high <= 1:
            raise Exception('stock[{name}] data too short'.format(name=self.name))
        self.idx = np.random.randint(low=1, high=high)
        self.step = 0
        self.actions.fill(0)
        self.navs.fill(1)

    def step(self, action):
        assert action in self.action_options, "%r (%s) invalid" % (action, type(action))
        # data step
        ###############################
        current_idx = self.idx + self.step
        visible_data = np.zeros((self.days, len(self.data.columns)))
        if self.step > 0:
            current_data = self.data[self.idx:current_idx].as_matrix()
            visible_data[:current_data.shape[0]] += current_data
        obs = visible_data
        nav_pct_change = self.pct_change.iat[current_idx, 3]  # close pct change
        done = bool(self.step >= self.days)

        # sim step
        ###############################
        reward = 0.0
        # record action
        self.actions[self.step] = action
        last_nav = 1.0 if self.step == 0 else self.navs[self.step-1]
        # record nav (只在LONG position情况下累积nav)
        self.navs[self.step] = last_nav * (nav_pct_change if action == 1 else 1.0)

        if not self.step == 0:
            # trading fee for changing trade position
            if abs(self.actions[self.step-1] - action) > 0:
                reward = self.navs[self.step] * self.trading_cost_pct_change
        if done:
            # episode finished, force sold
            reward = self.navs[self.step] * self.trading_cost_pct_change
        info = {
            'step': self.step,
            'reward': reward - 1.0,
            'nav': self.navs[self.step],
        }

        self.step += 1
        return obs, reward, done, info

    def snapshot(self):
        return {
            'idx': self.idx,
            'name': self.name,
            'days': self.days,
            'step': self.step,
            'actions': self.actions,
            'navs': self.navs,
        }

    def recover(self, snapshot, copy=True):
        assert(snapshot['name'] == self.name)
        self.idx = snapshot['idx']
        self.name = snapshot['name']
        self.days = snapshot['days']
        self.step = snapshot['step']
        self.actions = np.array(snapshot['actions'], copy=True) if copy else snapshot['actions']
        self.navs = np.array(snapshot['navs'], copy=True) if copy else snapshot['navs']
