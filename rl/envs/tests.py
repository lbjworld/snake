# coding: utf-8
import unittest
import timeit
import numpy as np

from envs.fast_trading_env import FastTradingEnv


class FastTradingEnvTestCase(unittest.TestCase):
    def setUp(self):
        self.days = 200
        self.env = FastTradingEnv(name='000333.SZ', days=self.days)

    def test_normal_run(self):
        self.assertTrue(self.env)
        episodes = 10
        for _ in range(episodes):
            self.env.reset()
            done = False
            count = 0
            while not done:
                action = np.random.choice(self.env.action_options(), 1)  # random
                observation, reward, done, info = self.env.step(action)
                self.assertTrue(reward >= 0.0)
                count += 1
        self.assertEqual(count, self.days)

    def test_performance(self):
        setup = """import numpy; from envs.fast_trading_env import FastTradingEnv; env = FastTradingEnv(name='000333.SZ', days=200)"""
        code = """env.reset()
done = False
while not done:
    action = numpy.random.choice(env.action_options(), 1)
    obs, reward, done, info = env.step(action)
"""
        count = 100
        total = timeit.timeit(code, setup=setup, number=count)
        print 'avg: {t} seconds'.format(t=total/count)

    def test_snapshot_recover(self):
        self.assertTrue(self.env)
        snapshot = None

        self.env.reset()
        done = False
        count = 0
        sum_reward = 0.0
        snapshot_sum_reward = 0.0
        while not done:
            action = 1  # fix to long
            observation, reward, done, info = self.env.step(action)
            count += 1
            sum_reward += reward
            if count == self.days / 2:
                # do snapshot
                snapshot = self.env.snapshot()
                snapshot_sum_reward = sum_reward
        self.assertTrue(snapshot)
        self.assertEqual(count, self.days)

        # recover to snapshot
        self.env.recover(snapshot)
        done = False
        count = 0
        recover_sum_reward = snapshot_sum_reward
        while not done:
            action = 1  # fix to long
            observation, reward, done, info = self.env.step(action)
            count += 1
            recover_sum_reward += reward
        self.assertEqual(count, self.days / 2)
        self.assertEqual(sum_reward, recover_sum_reward)

    def test_buy_hold_to_end(self):
        self.env.reset()
        done = False
        count = 0
        while not done:
            action = 1  # fix to long
            observation, reward, done, info = self.env.step(action)
            count += 1
            if not done:
                self.assertAlmostEqual(reward, 0.0)
            else:
                self.assertGreater(reward, 0.0)


if __name__ == '__main__':
    unittest.main()
