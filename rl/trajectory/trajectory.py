# coding: utf-8
from __future__ import unicode_literals

from gym_trading.envs import FastTradingEnv

from MCTS.mcts import MCTSBuilder
from MCTS.trading_policy import RandomTradingPolicy


class SimTrajectory(object):
    def __init__(self, env, model_policy, sim_policy, explore_rate=1e-01, debug=False):
        assert(env and model_policy and sim_policy)
        self._debug = debug
        self._main_env = env
        self._explore_rate = explore_rate
        self._exploit_rate = 1.0 - explore_rate
        self._exploit_policy = model_policy
        self._explore_policy = RandomTradingPolicy(action_options=self._main_env.action_options())
        self._sim_policy = sim_policy

        # change every step of trajectory
        self._sim_history = []

    @property
    def history(self):
        return self._sim_history

    def _state_evaluation(self, init_node=None, batch_size=100):
        tmp_env = FastTradingEnv(name=self._main_env.name, days=self._main_env.days)
        # do MCTS
        mcts_block = MCTSBuilder(tmp_env, init_node=init_node, debug=self._debug)
        root_node = mcts_block.run_batch(
            policies=[self._exploit_policy, self._explore_policy],
            probability_dist=[self._exploit_rate, self._explore_rate],
            env_snapshot=self._main_env.snapshot(),
            batch_size=batch_size
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
            'reward': reward,
        })
        return action, done

    def sim_run(self, batch_size=100):
        done = False
        init_node = None
        while not done:
            result_node = self._state_evaluation(init_node=init_node, batch_size=batch_size)
            action, done = self._sim_step(result_node.q_table)
            # set init_node to action node
            init_node = result_node._children[action]
        final_reward = self._sim_history[-1]['reward']  # last reward as final reward
        for item in self._sim_history:
            item['final_reward'] = final_reward
