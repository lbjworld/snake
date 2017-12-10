# coding: utf-8
from __future__ import unicode_literals

import logging
from concurrent import futures

logger = logging.getLogger(__name__)


def _init_model_func(model_dir, model_name, input_shape):
    from policy.resnet_trading_model import ResnetTradingModel
    # build model and save to model_dir
    model = ResnetTradingModel(
        name=model_name,
        model_dir=model_dir,
        load_model=False,
        input_shape=input_shape,
    )
    model_file_name = model.save_model(model_dir, model_name)
    return model_file_name


def _improve_func(
    model_dir, data_dir, model_name, input_shape, steps_per_epoch, batch_size, buffer_size
):
    from common.sim_dataset import SimDataSet
    from policy.resnet_trading_model import ResnetTradingModel
    # load `state of the art` model
    model = ResnetTradingModel(
        name=model_name,
        model_dir=model_dir,
        load_model=True,
        input_shape=input_shape,
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

    def __init__(self, data_dir, model_dir, input_shape, data_buffer_size=10000):
        self._input_shape = input_shape
        self._model_dir = model_dir
        self._data_dir = data_dir
        self._data_buffer_size = data_buffer_size

    def init_model(self, model_name):
        with futures.ProcessPoolExecutor(max_workers=1) as executor:
            f = executor.submit(
                _init_model_func, self._model_dir, model_name, self._input_shape
            )
            res = f.result()
            if not res:
                logger.error('init_model error:{e}'.format(e=f.exception()))
                return None
            return res

    def improve(self, model_name, batch_size=2048, steps_per_epoch=100):
        with futures.ProcessPoolExecutor(max_workers=1) as executor:
            f = executor.submit(
                _improve_func, self._model_dir, self._data_dir, model_name, self._input_shape,
                steps_per_epoch, batch_size, self._data_buffer_size,
            )
            new_model_file_name = f.result()
            if not new_model_file_name:
                logger.error('improve_model error:{e}'.format(e=f.exception()))
                return None
            return new_model_file_name
