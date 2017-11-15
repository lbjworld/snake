# coding: utf-8
from __future__ import unicode_literals

import unittest
from ete3 import Tree

from gym_trading.envs.fast_trading_env import FastTradingEnv
from trading_policy import RandomTradingPolicy
from trading_node import TradingNode


class RandomTradingPolicyTestCase(unittest.TestCase):
    def setUp(self):
        self.env = FastTradingEnv(name='000333.SZ', days=100)

    def test_trading_policy(self):
        action_options = self.env.action_options()
        policy = RandomTradingPolicy(action_options=action_options)
        self.assertTrue(policy)
        state = 'test state'
        action = policy.get_action(state)
        self.assertTrue(action in action_options)


class TradingNodeTestCase(unittest.TestCase):
    def setUp(self):
        self.stock_name = '000333.SZ'
        self.days = 30
        self.env = FastTradingEnv(name=self.stock_name, days=self.days)
        action_options = self.env.action_options()
        self.policy = RandomTradingPolicy(action_options=action_options)
        from utils import klass_factory
        self.TradingEnvNode = klass_factory(
            'Env_{name}_TradingNode'.format(name=self.stock_name),
            init_args={
                'env': self.env,
                'graph': Tree(),
            },
            base_klass=TradingNode
        )

    def run_one_episode(self, root_node, debug=False):
        self.assertTrue(self.env and self.policy and root_node)
        self.env.reset()
        current_node = root_node
        while current_node:
            if debug:
                print current_node._state, current_node._parent_action
            current_node = current_node.step(self.policy)
        return root_node

    def test_basic(self):
        self.assertTrue(self.TradingEnvNode)
        start_node = self.TradingEnvNode(state=None)
        root_node = self.run_one_episode(start_node, debug=True)
        self.assertTrue(root_node)
        self.assertTrue(start_node)
        self.assertEqual(root_node, start_node)
        self.assertEqual(root_node.get_rollout_count(), 1)
        root_node.show_graph()

    def test_multiple_episode(self):
        self.assertTrue(self.TradingEnvNode)
        count = 100
        root_node = self.TradingEnvNode(state=None)
        for i in range(count):
            root_node = self.run_one_episode(root_node)
        self.assertTrue(root_node)
        self.assertEqual(root_node.get_rollout_count(), count)
        self.assertEqual(root_node.visit_count, 0)
        top_visit_count = sum([c.visit_count for c in root_node._children])
        self.assertEqual(top_visit_count, count)
        root_node.show_graph()


if __name__ == '__main__':
    unittest.main()
