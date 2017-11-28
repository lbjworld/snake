# coding: utf-8
from __future__ import unicode_literals

import gc
from gym_trading.envs import FastTradingEnv

from MCTS.mcts import MCTSBuilder
from sim_policy import SimPolicy


class SimTrajectory(object):
    def __init__(self, env, model_policy, explore_rate=1e-01, debug=False):
        assert(env and model_policy)
        self._debug = debug
        self._main_env = env
        self._explore_rate = explore_rate
        self._exploit_policy = model_policy
        self._sim_policy = SimPolicy(action_options=self._main_env.action_options())

        # change every step of trajectory
        self._sim_history = []

    @property
    def history(self):
        return self._sim_history

    def __delete__(self):
        gc.collect()

    def _state_evaluation(self, init_node=None, rounds_per_step=100):
        # TODO: optimize when nearly end of episode, change from mcts to traverse search
        tmp_env = FastTradingEnv(name=self._main_env.name, days=self._main_env.days)
        # do MCTS
        mcts_block = MCTSBuilder(tmp_env, init_node=init_node, debug=self._debug)
        root_node = mcts_block.run_batch(
            policy=self._exploit_policy,
            env_snapshot=self._main_env.snapshot(),
            batch_size=rounds_per_step
        )
        return root_node

    def _sim_step(self, q_table):
        # get action with q_table
        action = self._sim_policy.get_action(q_table)
        # simulate on main env
        obs, reward, done, info = self._main_env.step(action)
        # save simluation history
        self._sim_history.append({
            'obs': obs,
            'q_table': q_table,
            'reward': reward,
        })
        return action, done

    def sim_run(self, rounds_per_step=100):
        done = False
        init_node = None
        while not done:
            result_node = self._state_evaluation(
                init_node=init_node, rounds_per_step=rounds_per_step)
            action, done = self._sim_step(result_node.q_table)
            # set init_node to action node
            init_node = result_node._down_edges[action]._down_node
        final_reward = self._sim_history[-1]['reward']  # last reward as final reward
        for item in self._sim_history:
            item['final_reward'] = final_reward
