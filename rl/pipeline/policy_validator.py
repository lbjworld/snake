# coding: utf-8
from __future__ import unicode_literals

import logging
import random
from concurrent import futures

from trajectory.sim_run import sim_run_func

logger = logging.getLogger(__name__)


class PolicyValidator(object):
    def __init__(
        self, episode_length, data_dir='./sim_data', model_dir='./models',
        tmp_model_dir='./tmp_models'
    ):
        self._episode_length = episode_length
        self._tmp_model_dir = tmp_model_dir
        self._model_dir = model_dir
        self._data_dir = data_dir

    def _validate_model(self, valid_stocks, model_dir, model_name, rounds=200, worker_num=4):
        # run sim trajectory on model, and return average reward
        _result = []
        with futures.ProcessPoolExecutor(max_workers=worker_num) as executor:
            future_to_idx = dict((executor.submit(sim_run_func, {
                'stock_name': random.choice(valid_stocks),
                'episode_length': self._episode_length,
                'rounds_per_step': 1000,
                'model_name': self._model_name,
                'model_dir': self._model_dir,
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

    def validate(self, valid_stocks, src, target):
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
        )
        target_avg_reward = self._validate_model(
            valid_stocks=valid_stocks,
            model_dir=self._tmp_model_dir,
            model_name=target,
        )
        return target if target_avg_reward > src_avg_reward else src
