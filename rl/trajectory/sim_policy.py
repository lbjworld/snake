# coding: utf-8
from __future__ import unicode_literals

from MCTS.base_policy import BasePolicy


class SimPolicy(BasePolicy):
    def __init__(self, action_options, debug=False):
        self.action_options = action_options
        self._debug = debug

    def get_action(self, state):
        assert(isinstance(state, dict))
        max_value = -1.0
        optimized_action = None
        # state should be a q_table
        for action, value in state.items():
            if value > max_value:
                max_value = value
                optimized_action = int(action)
        assert(optimized_action in self.action_options)
        return optimized_action
