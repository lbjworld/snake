# coding: utf-8
from __future__ import unicode_literals

import logging
import os
import pickle
import numpy as np
from concurrent import futures

logger = logging.getLogger(__name__)


def _init_model_func(model_dir, model_name, episode_length):
    from policy.resnet_trading_model import ResnetTradingModel
    # build model and save to model_dir
    model = ResnetTradingModel(
        name=model_name,
        model_dir=model_dir,
        load_model=False,
        episode_days=episode_length,
    )
    model.save_model(model_dir, model_name)
    return True


def _load_sim_data(model_name, data_dir, action_space_size, size=1000):
    # TODO: add size limit
    # load sim data from data_dir
    _data_files = []
    for root, dirs, files in os.walk(data_dir):
        for file_name in files:
            if model_name in file_name:
                _data_files.append(os.path.join(root, file_name))
    logger.debug('get sim data files size({s})'.format(s=len(_data_files)))
    _x, _y = [], []
    for file_path in _data_files:
        with open(file_path, 'r') as f:
            records = pickle.load(f)
            for r in records:
                _x.append(r['obs'])
                # TODO: how to deal with 'final_reward' ?
                action_values = [0.0] * action_space_size
                for k, v in r['q_table'].items():
                    action_values[int(k)] = v
                _y.append(action_values)
    logger.debug('sim data loaded, size({xs})'.format(xs=len(_x)))
    return np.array(_x), np.array(_y)


def _improve_func(
    model_dir, tmp_model_dir, data_dir, src, target, episode_length, batch_size, epochs,
    action_space_size,
):
    from policy.resnet_trading_model import ResnetTradingModel
    # load src model
    model = ResnetTradingModel(
        name=src,
        model_dir=model_dir,
        load_model=True,
        episode_days=episode_length
    )
    # load train data
    train_x, y = _load_sim_data(
        model_name=src, data_dir=data_dir, action_space_size=action_space_size
    )
    logger.debug('train_x:{xs}, y:{ys}'.format(xs=train_x.shape, ys=y.shape))
    # training
    model.fit(train_x, y, epochs=epochs, batch_size=batch_size)
    # save model in tmp_model_dir
    model.save_model(tmp_model_dir, target)
    return True


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
        with futures.ProcessPoolExecutor(max_workers=1) as executor:
            f = executor.submit(
                _init_model_func, self._model_dir, model_name, self._episode_length
            )
            res = f.result()
            if not res:
                logger.error('init_model error:{e}'.format(e=f.exception()))
                return False
            return True

    def improve(self, src, target, batch_size=32, epochs=100, action_space_size=2):
        with futures.ProcessPoolExecutor(max_workers=1) as executor:
            f = executor.submit(
                _improve_func, self._model_dir, self._tmp_model_dir, self._data_dir, src, target,
                self._episode_length, batch_size, epochs, action_space_size,
            )
            res = f.result()
            if not res:
                logger.error('improve_model error:{e}'.format(e=f.exception()))
                return False
            return True
