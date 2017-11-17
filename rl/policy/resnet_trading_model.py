# coding: utf-8
from __future__ import unicode_literals

import keras
from resnet import ResnetBuilder


class ResnetTradingModel(object):
    def __init__(self, name, episode_days=200, feature_num=5):
        self._name = name
        self._episode_days = episode_days
        self._feature_num = feature_num
        # input: (channel, row, col) -> (1, episode_days, ticker)
        # output: action probability
        self._model = ResnetBuilder.build_resnet_18((1, episode_days, feature_num), 2)
        ResnetBuilder.check_model(self._model, name=self._name)
        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self._model.compile(
            optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy']
        )

    def train_on_batch(self, x, y):
        assert(self._model)
        # TODO: add model checkpoint
        return self._model.train_on_batch(x, y)

    def predict(self, x, debug=False):
        assert(self._model)
        return self._model.predict(x, verbose=debug)
