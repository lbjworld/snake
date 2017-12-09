# coding: utf-8
from __future__ import unicode_literals

import logging
import numpy as np
import pandas as pd

from envs.fast_trading_env import FastTradingEnv

logger = logging.getLogger(__name__)


class Evaluator(object):

    def __init__(self, model_dir, input_shape):
        self._model_dir = model_dir
        self._input_shape = input_shape

    def evaluate_on_env(self, model, env):
        eval_history = []
        done = False
        last_obs = env.observations()
        while not done:
            p, v = model.predict(last_obs)
            action = np.argmax(p)
            obs, reward, done, _ = env.step(action)
            eval_history.append({
                'action_values': p,
                'predict_reward': v,
                'real_reward': reward,
                'pre_obs': last_obs,
                'post_obs': obs,
                'action': action,
            })
            last_obs = obs
        return eval_history

    def evaluate(self, basic_model, evaluate_model, valid_stocks, rounds):
        # load both models
        from policy.resnet_trading_model import ResnetTradingModel
        bm = ResnetTradingModel(
            name=basic_model,
            model_dir=self._model_dir,
            input_shape=self._input_shape,
            load_model=True,
            specific_model_name=basic_model
        )
        em = ResnetTradingModel(
            name=evaluate_model,
            model_dir=self._model_dir,
            input_shape=self._input_shape,
            load_model=True,
            specific_model_name=evaluate_model
        )
        # start evaluation
        _count = 0
        basic_avg_reward, evaluate_avg_reward = 0.0, 0.0
        while _count < rounds:
            stock_name = np.random.choice(valid_stocks, 1)[0]
            try:
                env = FastTradingEnv(
                    name=stock_name, days=self._input_shape[0], use_adjust_close=False
                )
            except Exception as e:
                logger.exception('env init error, {e}'.format(e=e))
                continue
            env_snapshot = env.snapshot()
            basic_evals = self.evaluate_on_env(bm, env)
            basic_avg_reward += basic_evals[-1]['real_reward']
            env.recover(env_snapshot)
            evaluate_evals = self.evaluate_on_env(em, env)
            evaluate_avg_reward += evaluate_evals[-1]['real_reward']
            _count += 1
        return basic_avg_reward / rounds, evaluate_avg_reward / rounds

    def show_plot(self, eval_history):
        import matplotlib.pyplot as plt
        # prepare data
        obs = pd.DataFrame(eval_history[-1]['post_obs'])
        obs.columns = ['open', 'high', 'low', 'close', 'volume']
        actions = [item['action'] for item in eval_history]
        signals = []
        for a1, a2 in zip([0] + actions, actions + [0]):
            if a1 == 1 and a2 == 0:
                # sell
                signals.append(2)
            elif a1 == 0 and a2 == 1:
                # buy
                signals.append(1)
            else:
                signals.append(0)
        signals = signals[:-1]
        buy_signals = [idx for idx, a in enumerate(signals) if a == 1]
        sell_signals = [idx for idx, a in enumerate(signals) if a == 2]
        print 'actions: ', actions
        print 'signals: ', signals
        print 'predict_reward: ', eval_history[-1]['predict_reward']
        print 'real_reward: ', eval_history[-1]['real_reward']

        # draw figure
        fig, axs = plt.subplots(2, 1, figsize=(16, 8))
        axs[0].set_ylabel('Volume')
        axs[0].plot(
            buy_signals, obs['volume'][buy_signals], 'b^',
            sell_signals, obs['volume'][sell_signals], 'rv',
        )
        axs[0].bar(obs['volume'].index, obs['volume'])

        axs[1].set_ylabel('Close Price')
        axs[1].plot(
            buy_signals, obs['close'][buy_signals], 'b^',
            sell_signals, obs['close'][sell_signals], 'rv',
            obs['close'].index, obs['close'], 'r-',
            obs['high'].index, obs['high'], 'g--',
            obs['low'].index, obs['low'], 'k--',
        )

        plt.show()
