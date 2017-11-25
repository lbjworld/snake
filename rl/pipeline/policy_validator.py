# coding: utf-8
from __future__ import unicode_literals

import logging
import os
import random
import shutil
from concurrent import futures

from policy.utils import get_latest_file
from trajectory.sim_run import sim_run_func

logger = logging.getLogger(__name__)


class PolicyValidator(object):
    def __init__(
        self, episode_length, explore_rate=1e-01, data_dir='./sim_data', model_dir='./models',
        tmp_model_dir='./tmp_models', debug=False
    ):
        self._episode_length = episode_length
        self._explore_rate = explore_rate
        self._tmp_model_dir = tmp_model_dir
        self._model_dir = model_dir
        self._data_dir = data_dir
        self._debug = debug

    def _validate_model(self, valid_stocks, model_dir, model_name, rounds, worker_num):
        # run sim trajectory on model, and return average reward
        _result = []
        with futures.ProcessPoolExecutor(max_workers=worker_num) as executor:
            future_to_idx = dict((executor.submit(sim_run_func, {
                'stock_name': random.choice(valid_stocks),
                'episode_length': self._episode_length,
                'rounds_per_step': 10,
                'model_name': model_name,
                'model_dir': model_dir,
                'model_feature_num': 5,
                'sim_explore_rate': self._explore_rate,
                'debug': self._debug,
            }), i) for i in range(rounds))
            for future in futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                if future.exception():
                    logger.error('validate Sim[{i}] error: {e}'.format(
                        i=idx, e=future.exception())
                    )
                    continue
                logger.info('validate Sim[{i}] finished'.format(i=idx))
                r = future.result()
                _result.append(r[-1]['final_reward'])
        return sum(_result) * 1.0 / len(_result)

    def validate(self, valid_stocks, src, target, rounds=200, worker_num=4):
        """
        Args:
            src(string): src model name
            target(string): tmp model name
        Return:
            (string): selected model name
        """
        # validate src model
        src_avg_reward = self._validate_model(
            valid_stocks=valid_stocks,
            model_dir=self._model_dir,
            model_name=src,
            rounds=rounds,
            worker_num=worker_num,
        )
        target_avg_reward = self._validate_model(
            valid_stocks=valid_stocks,
            model_dir=self._tmp_model_dir,
            model_name=target,
            rounds=rounds,
            worker_num=worker_num,
        )
        selected_model = src
        if target_avg_reward > src_avg_reward:
            selected_model = target
            # move model from tmp_model_dir to model_dir
            selected_model_file = get_latest_file(self._tmp_model_dir, selected_model)
            shutil.copy(os.path.join(self._tmp_model_dir, selected_model_file), self._model_dir)
        return selected_model
