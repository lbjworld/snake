# coding: utf-8
from __future__ import unicode_literals


class BaseTradingModel(object):
    """Base Trading Model Class"""

    def train_on_batch(self, x, y):
        assert(self._model)
        self._model.train_on_batch(x, y)

    def predict(self, x, debug=False):
        assert(self._model)
        return self._model.predict(x, verbose=debug)
