# coding: utf-8
from __future__ import unicode_literals

import os
import logging
from concurrent import futures

from common import settings
from common.utils import get_file_name, get_dir_list
from common.filelock import FileLock

from evaluation.evaluate_run import eval_run_func

logger = logging.getLogger(__name__)


class PolicyValidator(object):

    CURRENT_MODEL_FILE = settings.CURRENT_MODEL_FILE

    def __init__(self, model_dir, input_shape, debug=False):
        assert(model_dir and len(input_shape) == 2)
        self._input_shape = input_shape
        self._model_dir = model_dir
        self._debug = debug
        self._validated_models = set()

    def validate(self, basic_model, evaluate_model, valid_stocks, rounds=10):
        """
        Args:
            basic_model(string): base model name
            evaluate_model(string): target model name
            valid_stocks(list): validation stock name list
            rounds(int): evaluation rounds
        Return:
            (bool): whether evaluate_model improved
        """
        with futures.ProcessPoolExecutor(max_workers=1) as executor:
            f = executor.submit(
                eval_run_func, {
                    'model_dir': self._model_dir,
                    'basic_model': basic_model,
                    'evaluate_model': evaluate_model,
                    'input_shape': self._input_shape,
                    'rounds': rounds,
                    'valid_stocks': valid_stocks,
                }
            )
            res = f.result()
            if res:
            	BAR, EAR = res
            	logger.info('basic_avg_reward:{bar}, evaluate_avg_reward:{ear}'.format(
            	    bar=BAR, ear=EAR,
            	))
                self._validated_models.add(evaluate_model)
                return EAR > BAR
        return False

    def find_latest_model_name(self, interval_seconds=600):
        """watch model dir and return new added model name"""
        current_model_name = None
        if os.path.exists(self.CURRENT_MODEL_FILE):
            with FileLock(file_name=self.CURRENT_MODEL_FILE) as lock:
                with open(lock.file_name, 'r') as f:
                    current_model_name = f.read()
                    current_model_name = current_model_name.strip()
        file_paths = get_dir_list(self._model_dir)
        latest_model_name = get_file_name(file_paths[0])
        if latest_model_name in self._validated_models:
            # ignore valiated model
            return None
        if current_model_name and current_model_name != latest_model_name:
            # new model found
            return latest_model_name
        return None
