# coding: utf-8
from __future__ import unicode_literals

import numpy as np
import keras

from resnet import ResnetBuilder


class ResnetTradingModel(object):
    def __init__(self, name, episode_days=200, feature_num=5):
        self._name = name
        self._episode_days = episode_days
        self._feature_num = feature_num
        # input: (channel, row, col) -> (1, episode_days, ticker)
        # output: action probability
        self._model = ResnetBuilder.build_resnet_18(
            input_shape=(1, episode_days, feature_num),
            num_outputs=2
        )
        ResnetBuilder.check_model(self._model, name=self._name)
        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self._model.compile(
            optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy']
        )

    def _preprocess(self, x):
        # extend to add channel dimension
        return np.expand_dims(x, axis=0)

    def train_on_batch(self, x, y):
        assert(self._model)
        # TODO: add model checkpoint
        return self._model.train_on_batch(self._preprocess(x), y)

    def predict(self, x, debug=False):
        assert(self._model)
        return self._model.predict(self._preprocess(x), verbose=debug)
