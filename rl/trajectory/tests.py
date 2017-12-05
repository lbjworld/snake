# coding: utf-8
from __future__ import unicode_literals

import unittest
from envs.fast_trading_env import FastTradingEnv

from policy.resnet_trading_model import ResnetTradingModel
from policy.model_policy import ModelTradingPolicy

from trajectory.sim_trajectory import SimTrajectory


class SimTrajectoryTestCase(unittest.TestCase):
    def setUp(self):
        self.days = 30
        self.env = FastTradingEnv(name='000333.SZ', days=self.days)
        self.env.reset()

    def test_run_trajectory(self):
        self.assertTrue(self.env)
        resnet_model = ResnetTradingModel(
            name='test_resnet_model',
            model_dir='./test_models',
            input_shape=(self.days, 5),  # open, high, low, close, volume
        )
        self.assertTrue(resnet_model)
        model_policy = ModelTradingPolicy(
            action_options=self.env.action_options(),
            model=resnet_model,
            debug=False
        )
        self.assertTrue(model_policy)

        # start trajectory
        t = SimTrajectory(
            env=self.env,
            model_policy=model_policy,
            debug=False,
        )
        t.sim_run(rounds_per_step=23)
        print t.history
        self.assertEqual(len(t.history), self.days)


if __name__ == '__main__':
    unittest.main()
