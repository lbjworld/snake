# coding: utf-8
from __future__ import unicode_literals

import os
import time
import numpy as np
import keras

from resnet import ResnetBuilder
from utils import get_latest_file


class ResnetTradingModel(object):
    def __init__(
        self, name, model_dir='./models', load_model=False,
        episode_days=200, feature_num=5
    ):
        self._model_dir = model_dir
        self._episode_days = episode_days
        self._feature_num = feature_num
        assert(name)
        self._name = name
        if load_model:
            # load from model dir
            self._model = self._load_latest_model(
                name=name, model_dir=self._model_dir
            )
        else:
            # build from scratch
            self._model = self._build_model(name=name)
        assert(self._model)

    def _build_model(self, name):
        """build model from scratch"""
        # input: (channel, row, col) -> (1, episode_days, ticker)
        # output: action probability
        _model = ResnetBuilder.build_resnet_18(
            input_shape=(1, self._episode_days, self._feature_num),
            num_outputs=2
        )
        ResnetBuilder.check_model(_model, name=name)
        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        _model.compile(
            optimizer=adam, loss='mean_squared_error', metrics=['accuracy']
        )
        return _model

    def _load_latest_model(self, model_dir, name):
        """load latest model by name"""
        latest_name = get_latest_file(model_dir, name)
        if not latest_name:
            raise Exception(
                '{k} latest model not found.'.format(k=self.__klass__))
        model_path = os.path.join(model_dir, latest_name)
        assert(model_path)
        _model = self._build_model(name)
        _model.load_weights(model_path)
        return _model

    def _gen_model_name(self, name):
        return '{n}.{ts}.h5'.format(n=name, ts=int(time.time()))

    def save_model(self, model_dir, name):
        assert(self._model)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, self._gen_model_name(name))
        self._model.save_weights(model_path)

    def _preprocess(self, batch_x):
        # extend to add channel dimension
        return np.expand_dims(batch_x, axis=3)

    def train_on_batch(self, batch_x, y):
        assert(self._model)
        return self._model.train_on_batch(self._preprocess(batch_x), y)

    def fit(self, train_x, y, epochs, batch_size=32):
        assert(self._model)
        return self._model.fit(
            self._preprocess(train_x), y, epochs=epochs, batch_size=batch_size,
            shuffle=True, validation_split=0.1
        )

    def predict(self, x, debug=False):
        assert(self._model)
        batch_x = np.expand_dims(x, axis=0)
        return self._model.predict(self._preprocess(batch_x), batch_size=1, verbose=debug)
