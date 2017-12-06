# coding: utf-8
from __future__ import unicode_literals

import logging
import os
import time
import numpy as np
# from sklearn import preprocessing

from common import settings
from common.filelock import FileLock

logger = logging.getLogger(__name__)


class ResnetTradingModel(object):
    CURRENT_MODEL_FILE = settings.CURRENT_MODEL_FILE

    def __init__(self, name, model_dir, input_shape, load_model=False, specific_model_name=None):
        assert(name and len(input_shape) == 2)
        self._model_dir = model_dir
        self._episode_days, self._feature_num = input_shape
        self._name = name
        self._specific_model_name = specific_model_name
        if load_model:
            if not self._specific_model_name:
                # load current latest from model dir
                self._model = self._load_latest_model(model_dir=self._model_dir)
            else:
                # load specific model with `specific_model_name`
                self._model = self._load_model(
                    name=self._specific_model_name, model_dir=self._model_dir
                )
        else:
            # build from scratch
            self._model = self._build_model(name=name)
        assert(self._model)

    def _build_model(self, name, optimizer=None):
        from resnet import ResnetBuilder
        """build model from scratch"""
        # input: (channel, row, col) -> (1, episode_days, ticker)
        # output: action probability
        _model = ResnetBuilder.build_resnet_18(
            input_shape=(1, self._episode_days, self._feature_num),
            num_outputs=2
        )
        logger.debug('built trade model[{name}]'.format(name=name))
        # ResnetBuilder.check_model(_model, name=name)
        if not optimizer:
            import keras
            optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9)
        _model.compile(
            optimizer=optimizer,
            loss={
                'policy_header': 'cosine_proximity',
                'value_header': 'mean_squared_error',
            },
            metrics={
                'policy_header': ['mse', 'acc'],
                'value_header': ['mse'],
            },
        )
        logger.debug('compiled trade model[{name}]'.format(name=name))
        return _model

    def _load_latest_model(self, model_dir):
        """load latest model by name"""
        with FileLock(file_name=self.CURRENT_MODEL_FILE) as lock:
            with open(lock.file_name, 'r') as f:
                latest_name = f.read().strip()
                model_path = os.path.join(model_dir, latest_name)
                _model = self._build_model(latest_name)
                logger.debug('loading trade model from [{p}]'.format(p=model_path))
                _model.load_weights(model_path)
                return _model

    def _load_model(self, model_dir, name):
        _model = self._build_model(name)
        model_path = os.path.join(model_dir, name)
        logger.debug('loading specific trade model from [{p}]'.format(p=model_path))
        _model.load_weights(model_path)
        return _model

    def _gen_model_name(self, name):
        return '{n}.{ts}.h5'.format(n=name, ts=int(time.time()))

    def save_model(self, model_dir, name):
        assert(self._model)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        new_model_name = self._gen_model_name(name)
        model_path = os.path.join(model_dir, new_model_name)
        self._model.save_weights(model_path)
        logger.debug('save trade model[{name}] to [{p}]'.format(name=name, p=model_path))
        return new_model_name

    def _preprocess(self, batch_x):
        # l2 normalize
        # for i in range(batch_x.shape[0]):
        #    batch_x[i] = preprocessing.normalize(batch_x[i], norm='l2')
        # extend to add channel dimension
        return np.expand_dims(batch_x, axis=3)

    def train_on_batch(self, batch_x, y):
        return self._model.train_on_batch(self._preprocess(batch_x), y)

    def fit(self, train_x, y, epochs, batch_size=32):
        return self._model.fit(
            self._preprocess(train_x), y, epochs=epochs, batch_size=batch_size,
            shuffle=True,  # validation_split=0.1
        )

    def fit_generator(self, generator, steps_per_epoch):
        return self._model.fit_generator(generator=generator, steps_per_epoch=steps_per_epoch)

    def predict(self, x, debug=False):
        batch_x = np.expand_dims(x, axis=0)
        outputs = self._model.predict(self._preprocess(batch_x), batch_size=1, verbose=debug)
        return np.squeeze(outputs[0], axis=0), np.squeeze(outputs[1], axis=0)[0]
