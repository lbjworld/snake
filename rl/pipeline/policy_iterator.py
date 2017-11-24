# coding: utf-8
from __future__ import unicode_literals

import logging
import os
import pickle
import numpy as np
from collections import OrderedDict

from policy.resnet_trading_model import ResnetTradingModel

logger = logging.getLogger(__name__)


class PolicyIterator(object):

    def __init__(
        self, episode_length, data_dir='./sim_data', model_dir='./models',
        tmp_model_dir='./tmp_models'
    ):
        self._episode_length = episode_length
        self._tmp_model_dir = tmp_model_dir
        self._model_dir = model_dir
        self._data_dir = data_dir

    def init_model(self, model_name):
        # build model and save to model_dir
        model = ResnetTradingModel(
            name=model_name,
            model_dir=self._model_dir,
            load_model=False,
            episode_days=self._episode_length,
        )
        model.save_model(self._model_dir, model_name)

    def _load_sim_data(self, model_name, size=1000):
        # TODO: add size limit
        # load sim data from data_dir
        _data_files = []
        for root, dirs, files in os.walk(self._data_dir):
            for file_name in files:
                if model_name in file_name:
                    _data_files.append(os.path.join(root, file_name))
        _x, _y = [], []
        for file_path in _data_files:
            with open(file_path, 'r') as f:
                records = pickle.load(f)
                for r in records:
                    _x.append(r['obs'])
                    # TODO: how to deal with 'final_reward' ?
                    _y.append(
                        OrderedDict(sorted(r['q_table'].items())).values()
                    )
        return np.array(_x), np.array(_y)

    def improve(self, src, target, batch_size=32, epochs=100):
        # load src model
        model = ResnetTradingModel(
            name=src,
            model_dir=self._model_dir,
            load_model=True,
            episode_days=self._episode_length
        )
        # load train data
        train_x, y = self._load_sim_data(model_name=src)
        # training
        model.fit(train_x, y, epochs=epochs, batch_size=batch_size)
        # save model in tmp_model_dir
        model.save_model(self._tmp_model_dir, target)
