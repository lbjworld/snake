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
        data_df = data_df[(~np.isnan(data_df.volume)) & (data_df.volume > 1e-9)]  # 跳过所有停牌日
        self.pct_change = data_df.pct_change().fillna(0.0) + 1.0  # 计算变化量
        data_df.volume = data_df.volume / FastTradingEnv.VOLUME_SCALE_FACTOR
        self.data = data_df.as_matrix()

        self.trading_cost_pct_change = 1.0 - trading_cost_bps
        self._actions = np.zeros(self.days)
        self._navs = np.ones(self.days)

        self.reset()

    @property
    def action_space(self):
        return [0, 1]

    def action_options(self):
        return [0, 1]

    def reset(self):
        # we want continuous data
        high = self.data.shape[0] - self.days
        if high <= 1:
            raise Exception('stock[{name}] data too short'.format(name=self.name))
        self._idx = np.random.randint(low=1, high=high)
        self._step = 0
        self._actions.fill(0)
        self._navs.fill(1)

    def step(self, action):
        assert action in self.action_space, "%r (%s) invalid" % (action, type(action))
        # data step
        ###############################
        _next_step = self._step + 1
        # get next obs
        obs = self.observations(_next_step)
        # close pct change
        nav_pct_change = self.pct_change.iat[self._idx + _next_step, 3]
        done = bool(_next_step >= self.days)

        # sim step
        ###############################
        reward = 0.0
        # record action
        self._actions[self._step] = action
        last_nav = 1.0 if self._step == 0 else self._navs[self._step-1]
        # record nav (只在LONG position情况下累积nav)
        self._navs[self._step] = last_nav * (nav_pct_change if action == 1 else 1.0)

        if not self._step == 0:
            # trading fee for changing trade position
            if abs(self._actions[self._step-1] - action) > 0:
                reward = self._navs[self._step] * self.trading_cost_pct_change - 1.0
        if done:
            # episode finished, force sold
            reward = self._navs[self._step] * self.trading_cost_pct_change - 1.0
        info = {
            'step': self._step,
            'reward': reward,
            'nav': self._navs[self._step],
        }
        # TODO: add shortcut for low NAV

        self._step += 1
        return obs, reward, done, info

    def observations(self, data_step=0):
        """get current observations"""
        current_idx = self._idx + data_step
        obs = np.zeros((self.days, self.data.shape[1]))
        if data_step > 0:
            current_data = self.data[self._idx:current_idx, :]
            obs[:current_data.shape[0]] += current_data
        return obs

    def snapshot(self):
        return {
            'idx': self._idx,
            'name': self.name,
            'days': self.days,
            'step': self._step,
            'actions': self._actions,
            'navs': self._navs,
        }

    def recover(self, snapshot, copy=True):
        assert(snapshot['name'] == self.name)
        self._idx = snapshot['idx']
        self.name = snapshot['name']
        self.days = snapshot['days']
        self._step = snapshot['step']
        self._actions = np.array(snapshot['actions'], copy=True) if copy else snapshot['actions']
        self._navs = np.array(snapshot['navs'], copy=True) if copy else snapshot['navs']
