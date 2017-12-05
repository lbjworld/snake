# coding: utf-8
from __future__ import unicode_literals

import unittest

from envs.fast_trading_env import FastTradingEnv
from MCTS.mcts import MCTSBuilder
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
            model_dir='test_models',
            input_shape=(self.days, 5),  # open, high, low, close, volume
        )
        self.assertTrue(resnet_model)
        exploit_policy = ModelTradingPolicy(
            action_options=self.env.action_options(),
            model=resnet_model,
            debug=False
        )
        self.assertTrue(exploit_policy)
        # init env and save snapshot
        self.env.reset()
        snapshot_v0 = self.env.snapshot()
        # init mcts block
        mcts_block = MCTSBuilder(self.env, debug=True)
        mcts_block.clean_up()
        # run batch and get q_table of next step
        root_node = mcts_block.run_batch(
            policy=exploit_policy,
            env_snapshot=snapshot_v0,
            batch_size=50
        )
        self.assertTrue(root_node)
        root_node.show_graph(name='model_policy_tree')
        q_table = root_node.q_table
        self.assertTrue(q_table)
        print q_table
        print root_node.show_final_state()


if __name__ == '__main__':
    unittest.main()
