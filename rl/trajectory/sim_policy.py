# coding: utf-8
from __future__ import unicode_literals

import numpy as np

from MCTS.base_policy import BasePolicy


class SimPolicy(BasePolicy):
    def __init__(self, action_options, debug=False):
        assert(isinstance(action_options, list))
        self.action_options = action_options
        self._debug = debug

    def get_action(self, state):
        # state should be a q_table
        return np.argmax(state)

    def evaluate(self, state):
        pass
