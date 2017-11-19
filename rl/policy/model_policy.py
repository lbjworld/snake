# coding: utf-8
from __future__ import unicode_literals

import numpy as np

from MCTS.base_policy import BasePolicy


class ModelTradingPolicy(BasePolicy):
    def __init__(self, action_options, model=None, debug=False):
        self.action_options = action_options
        assert(model)
        self._model = model
        self._debug = debug

    def get_action(self, state):
        assert(isinstance(self.action_options, list))
        if not np.any(state):
            # init state
            return self.action_options[0]
        # predict by using model
        predict_actions = self._model.predict(state, debug=self._debug)
        return np.argmax(predict_actions)
