# coding: utf-8
from __future__ import unicode_literals

import unittest
from gym_trading.envs import FastTradingEnv

from MCTS.mcts import MCTSBuilder
from MCTS.trading_policy import RandomTradingPolicy
from resnet_trading_model import ResnetTradingModel
from model_policy import ModelTradingPolicy


class ModelPolicyTestCase(unittest.TestCase):
    def setUp(self):
        self.days = 30
        self.env = FastTradingEnv(name='000333.SZ', days=self.days)

    def test_usage_sample(self):
        self.assertTrue(self.env)
        resnet_model = ResnetTradingModel(
            name='test_resnet_model',
            episode_days=self.days,
            feature_num=5,  # open, high, low, close, volume
        )
        self.assertTrue(resnet_model)
        exploit_policy = ModelTradingPolicy(
            action_options=self.env.action_options(),
            model=resnet_model,
            debug=False
        )
        self.assertTrue(exploit_policy)
        explore_policy = RandomTradingPolicy(action_options=self.env.action_options())
        # init env and save snapshot
        self.env.reset()
        snapshot_v0 = self.env.snapshot()
        # init mcts block
        mcts_block = MCTSBuilder(self.env, debug=True)
        mcts_block.clean_up()
        # run batch and get q_table of next step
        root_node = mcts_block.run_batch(
            policies=[exploit_policy, explore_policy],
            probability_dist=[0.9, 0.1],
            env_snapshot=snapshot_v0,
            batch_size=50
        )
        root_node = root_node._children[0]
        self.assertTrue(root_node)
        root_node.show_graph()
        q_table = root_node.q_table
        print q_table
        self.assertTrue(q_table)


if __name__ == '__main__':
    unittest.main()
