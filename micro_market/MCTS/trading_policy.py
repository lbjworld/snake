# coding: utf-8
from __future__ import unicode_literals

import random

from base_policy import BasePolicy


class RandomTradingPolicy(BasePolicy):
    def __init__(self, action_options):
        self.action_options = action_options

    def get_action(self, state):
        assert(isinstance(self.action_options, list))
        return random.choice(self.action_options)

    def evaluate(self, state):
        action_size = len(self.action_options)
        p = 1.0 / action_size
        return [p] * action_size, random.uniform(0.0, 1.0)


class HoldTradingPolicy(BasePolicy):
    def __init__(self, action_options, action_idx=0):
        self.action_options = action_options
        self.action_idx = action_idx

    def get_action(self, state):
        assert(isinstance(self.action_options, list))
        return self.action_options[self.action_idx]

    def evaluate(self, state):
        action_size = len(self.action_options)
        p = 1.0 / action_size
        return [p] * action_size, 1.0

