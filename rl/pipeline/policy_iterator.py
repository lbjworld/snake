# coding: utf-8
from __future__ import unicode_literals

import logging
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


def _improve_func(
    model_dir, data_dir, model_name, episode_length, steps_per_epoch, batch_size, buffer_size
):
    from common.sim_dataset import SimDataSet
    from policy.resnet_trading_model import ResnetTradingModel
    # load `state of the art` model
    model = ResnetTradingModel(
        name=model_name,
        model_dir=model_dir,
        load_model=True,
        episode_days=episode_length
    )
    # load train data
    sim_ds = SimDataSet(data_dir=data_dir, pool_size=buffer_size)
    current_model_file_name = None
    while True:
        # training forever
        model.fit_generator(
            generator=sim_ds.generator(batch_size=batch_size),
            steps_per_epoch=steps_per_epoch,
        )
        # checkpoint: save model in model_dir
        current_model_file_name = model.save_model(model_dir, model_name)
    return current_model_file_name


class PolicyIterator(object):

    def __init__(
        self, episode_length, data_dir='./sim_data', model_dir='./models', data_buffer_size=10000,
    ):
        self._episode_length = episode_length
        self._model_dir = model_dir
        self._data_dir = data_dir
        self._data_buffer_size = data_buffer_size

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

    def improve(self, model_name, batch_size=2048, steps_per_epoch=100):
        with futures.ProcessPoolExecutor(max_workers=1) as executor:
            f = executor.submit(
                _improve_func, self._model_dir, self._data_dir, model_name, self._episode_length,
                steps_per_epoch, batch_size, self._data_buffer_size,
            )
            new_model_file_name = f.result()
            if not new_model_file_name:
                logger.error('improve_model error:{e}'.format(e=f.exception()))
                return None
            return new_model_file_name
