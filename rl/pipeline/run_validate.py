# coding: utf-8
from __future__ import unicode_literals

import time
import logging

from common.dataset import StockDataSet
from common.utils import set_current_model
from common import settings
from pipeline.policy_validator import PolicyValidator

logger = logging.getLogger(__name__)


def validation():
    """
        1. watch model dir
        2. when new model added, do validation
        3. update current model file if new added model is better then old one
    """
    ds = StockDataSet()
    stock_codes = ds.stock_list(min_days=settings.EPISODE_LENGTH)
    policy_validator = PolicyValidator(
        model_dir=settings.MODEL_DATA_DIR,
        input_shape=(settings.EPISODE_LENGTH, settings.FEATURE_NUM),
    )
    base_model_name = None
    while True:
        new_model_name = policy_validator.find_latest_model_name()
        if new_model_name:
            logger.info('[VALIDATION] found new model[{mn}]'.format(mn=new_model_name))
            # get base model name
            with open(settings.CURRENT_MODEL_FILE, 'r') as f:
                base_model_name = f.read().strip()
            assert(base_model_name)
            # policy validation (compare between target and src)
            improved = policy_validator.validate(
                basic_model=base_model_name,
                evaluate_model=new_model_name,
                valid_stocks=stock_codes[ds.TRAIN_SIZE:ds.TRAIN_SIZE+ds.VALID_SIZE],
                rounds=settings.VALID_ROUNDS,
            )
            if improved:
                set_current_model(model_name=new_model_name, model_dir=settings.MODEL_DATA_DIR)
                logger.info(
                    '[VALIDATION] model improved, new model[{mn}]'.format(mn=new_model_name)
                )
            else:
                logger.info(
                    '[VALIDATION] model improve failed, model[{mn}]'.format(mn=new_model_name)
                )
        else:
            logger.info('[VALIDATION] new model not found, waiting ...')
        time.sleep(settings.VALID_INTERVAL)


if __name__ == '__main__':
    assert(logger)
    logging.basicConfig(filename='validation.log', level=logging.INFO)
    validation()
