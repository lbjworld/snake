# coding: utf-8
from __future__ import unicode_literals

import random

from base_policy import BasePolicy


class TradingPolicy(BasePolicy):
    def __init__(self, action_options):
        self.action_options = action_options

    def get_action(self, state):
        assert(isinstance(self.action_options, list))
        # choose action by considering state
        # TODO: random first
        return random.choice(self.action_options)
