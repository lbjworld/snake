# coding: utf-8
from __future__ import unicode_literals

import logging
import os
import pickle
import time
import random
from tqdm import tqdm
from concurrent import futures

from trajectory.sim_run import sim_run_func

logger = logging.getLogger(__name__)


class SimGenerator(object):
    """generate simulate data"""

    def __init__(
        self, train_stocks, model_name, explore_rate, input_shape, model_dir,
        data_dir, debug=False, sim_count=2500, rounds_per_step=1000, worker_timeout=300,
    ):
        assert(len(input_shape) == 2)
        self._model_name = model_name
        self._model_dir = model_dir
        self._input_shape = input_shape
        self._explore_rate = explore_rate
        self._train_stocks = train_stocks
        self._data_dir = data_dir
        self._sim_count = sim_count
        self._rounds_per_step = rounds_per_step
        self._worker_timeout = worker_timeout
        self._debug = debug

    def _get_sim_file_path(self, data_dir):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        file_name = '{mn}.{ts}.pkl'.format(mn=self._model_name, ts=int(time.time()))
        return os.path.join(data_dir, file_name)

    def run(self, sim_batch_size=100, worker_num=4):
        episode_length = self._input_shape[0]
        batch_data_length = sim_batch_size * episode_length
        progress_bar = tqdm(total=self._sim_count)
        for idx in range(0, self._sim_count, sim_batch_size):
            _results = []
            while len(_results) < batch_data_length:
                with futures.ProcessPoolExecutor(max_workers=worker_num) as executor:
                    _tasks = [executor.submit(sim_run_func, {
                        'stock_name': random.choice(self._train_stocks),
                        'input_shape': self._input_shape,
                        'rounds_per_step': self._rounds_per_step,
                        'model_name': self._model_name,
                        'model_dir': self._model_dir,
                        'sim_explore_rate': self._explore_rate,
                        'debug': self._debug,
                    }) for i in range(worker_num)]
                    try:
                        for future in futures.as_completed(_tasks, timeout=self._worker_timeout):
                            exception = future.exception()
                            if exception:
                                logger.error('[SIM] one sim error: {e}'.format(e=exception))
                                continue
                            logger.info('[SIM] one sim finished')
                            _results.extend(future.result())
                            progress_bar.update(1)
                    except futures.TimeoutError:
                        logger.error('[SIM] some futures timeout')

            # save results to file
            _results = _results[:batch_data_length]
            file_path = self._get_sim_file_path(self._data_dir)
            with open(file_path, 'w') as f:
                pickle.dump(_results, f)

        progress_bar.close()
