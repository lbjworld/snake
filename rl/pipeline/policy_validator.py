# coding: utf-8
from __future__ import unicode_literals

import os
import logging
import random
from concurrent import futures

from common import settings
from common.filelock import FileLock
from trajectory.sim_run import sim_run_func

logger = logging.getLogger(__name__)


class PolicyValidator(object):

    CURRENT_MODEL_FILE = settings.CURRENT_MODEL_FILE

    def __init__(
        self, episode_length, explore_rate=1e-01, data_dir='./sim_data', model_dir='./models',
        debug=False
    ):
        self._episode_length = episode_length
        self._explore_rate = explore_rate
        self._model_dir = model_dir
        self._data_dir = data_dir
        self._debug = debug
        self._validated_models = []

    def _validate_model(
        self, valid_stocks, model_dir, model_name, rounds, rounds_per_step, worker_num,
        specific_model_name=None
    ):
        # run sim trajectory on model, and return average reward
        _result = []
        for i in range(0, rounds, worker_num):
            with futures.ProcessPoolExecutor(max_workers=worker_num) as executor:
                future_to_idx = dict((executor.submit(sim_run_func, {
                    'stock_name': random.choice(valid_stocks),
                    'episode_length': self._episode_length,
                    'rounds_per_step': rounds_per_step,
                    'model_name': model_name,
                    'model_dir': model_dir,
                    'model_feature_num': 5,
                    'sim_explore_rate': self._explore_rate,
                    'specific_model_name': specific_model_name,
                    'debug': self._debug,
                }), i+j) for j in range(worker_num))
                for future in futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    if future.exception():
                        logger.error('validate Sim[{idx}] error: {e}'.format(
                            idx=idx, e=future.exception())
                        )
                        continue
                    logger.info('validate Sim[{idx}] finished'.format(idx=idx))
                    r = future.result()
                    _result.append(r[-1]['final_reward'])
        if not _result:
            return 0.0
        return sum(_result) * 1.0 / len(_result)

    def validate(self, valid_stocks, base, target, rounds=200, rounds_per_step=100, worker_num=4):
        """
        Args:
            base(string): base model name
            target(string): tmp model file name
        Return:
            (string): selected model name
        """
        # validate src model
        src_avg_reward = self._validate_model(
            valid_stocks=valid_stocks,
            model_dir=self._model_dir,
            model_name=base,
            rounds=rounds,
            rounds_per_step=rounds_per_step,
            worker_num=worker_num,
        )
        target_avg_reward = self._validate_model(
            valid_stocks=valid_stocks,
            model_dir=self._model_dir,
            model_name=base,
            rounds=rounds,
            rounds_per_step=rounds_per_step,
            worker_num=worker_num,
            specific_model_name=target,
        )
        logger.info('src_avg_reward:{sar}, target_avg_reward:{tar}'.format(
            sar=src_avg_reward, tar=target_avg_reward,
        ))
        return target_avg_reward > src_avg_reward

    def find_latest_model_name(self, interval_seconds=600):
        """watch model dir and return new added model name"""
        current_model_path = None
        if os.path.exists(self.CURRENT_MODEL_FILE):
            with FileLock(file_name=self.CURRENT_MODEL_FILE) as lock:
                with open(lock.file_name, 'r') as f:
                    current_model_name = f.read()
                    current_model_path = os.path.join(self._model_dir, current_model_name)
        file_paths = os.listdir(self._model_dir)
        file_paths = [os.path.join(self._model_dir, fp) for fp in file_paths]
        file_paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)  # new -> old
        if current_model_path != file_paths[:1]:
            # new model found
            head, tail = os.path.split(file_paths[:1])
            return tail
        return None
