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
    model_file_name = model.save_model(model_dir, model_name)
    return model_file_name


def _load_sim_data(model_name, data_dir, target_reward, size=10000):
    # TODO: add size limit
    # load sim data from data_dir
    _data_files = []
    for root, dirs, files in os.walk(data_dir):
        for file_name in files:
            if model_name in file_name:
                _data_files.append(os.path.join(root, file_name))
    logger.debug('get sim data files size({s})'.format(s=len(_data_files)))
    _x, p_y, v_y = [], [], []
    for file_path in _data_files:
        with open(file_path, 'r') as f:
            records = pickle.load(f)
            np.random.shuffle(records)
            for r in records:
                _x.append(r['obs'])
                p = r['q_table']
                p_y.append(p)
                v = r['final_reward'] - target_reward
                v_y.append(v)
    logger.debug('sim data loaded, size({xs})'.format(xs=len(_x)))
    random_indices = np.random.choice(range(len(_x)), size)
    logger.debug('random choose size({s})'.format(s=size))
    return (
        np.take(_x, random_indices, axis=0),
        [np.take(p_y, random_indices, axis=0), np.take(v_y, random_indices, axis=0)]
    )


def _improve_func(
    model_dir, data_dir, model_name, episode_length, batch_size, epochs,
    target_reward, total_size=3000,
):
    from policy.resnet_trading_model import ResnetTradingModel
    # load src model
    model = ResnetTradingModel(
        name=model_name,
        model_dir=model_dir,
        load_model=True,
        episode_days=episode_length
    )
    # load train data
    train_x, y = _load_sim_data(
        model_name=model_name, data_dir=data_dir, target_reward=target_reward, size=total_size
    )
    logger.debug(
        'train_x:{xs}, policy_y:{pys}, value_y:{vys}'.format(
            xs=train_x.shape, pys=y[0].shape, vys=y[1].shape
        )
    )
    # training
    # TODO: fix here, use batch train
    model.fit(train_x, y, epochs=epochs, batch_size=batch_size)
    # save model in model_dir
    model_file_name = model.save_model(model_dir, model_name)
    return model_file_name


class PolicyIterator(object):

    def __init__(
        self, episode_length, data_dir='./sim_data', model_dir='./models', target_reward=1.0,
    ):
        self._episode_length = episode_length
        self._model_dir = model_dir
        self._data_dir = data_dir
        self._target_reward = target_reward

    def init_model(self, model_name):
        with futures.ProcessPoolExecutor(max_workers=1) as executor:
            f = executor.submit(
                _init_model_func, self._model_dir, model_name, self._episode_length
            )
            res = f.result()
            if not res:
                logger.error('init_model error:{e}'.format(e=f.exception()))
                return None
            return res

    def improve(self, model_name, batch_size=32, epochs=100):
        with futures.ProcessPoolExecutor(max_workers=1) as executor:
            f = executor.submit(
                _improve_func, self._model_dir, self._data_dir, model_name,
                self._episode_length, batch_size, epochs, self._target_reward,
            )
            new_model_file_name = f.result()
            if not new_model_file_name:
                logger.error('improve_model error:{e}'.format(e=f.exception()))
                return None
            return new_model_file_name
