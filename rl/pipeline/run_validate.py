# coding: utf-8
from __future__ import unicode_literals

import logging

from common.dataset import StockDataSet
from common.utils import set_current_model
from pipeline.policy_validator import PolicyValidator

logger = logging.getLogger(__name__)


EPISODE_LENGTH = 30
VALID_ROUNDS = 100
VALID_ROUNDS_PER_STEP = 23

CPU_CORES = 2
CURRENT_MODEL_FILE = 'model.current'


def validation(base_model_name):
    """
        1. watch model dir
        2. when new model added, do validation
        3. update current model file if new added model is better then old one
    """
    ds = StockDataSet()
    stock_codes = ds.stock_list(min_days=EPISODE_LENGTH)
    policy_validator = PolicyValidator(episode_length=EPISODE_LENGTH)
    while True:
        new_model_name = policy_validator.find_latest_model_name()
        # policy validation (compare between target and src)
        improved = policy_validator.validate(
            valid_stocks=stock_codes[ds.TRAIN_SIZE:ds.TRAIN_SIZE+ds.VALID_SIZE],
            base=base_model_name,
            target=new_model_name,
            rounds=VALID_ROUNDS,
            rounds_per_step=VALID_ROUNDS_PER_STEP,
            worker_num=CPU_CORES,
        )
        if improved:
            set_current_model(model_name=new_model_name)
            logger.info('[VALIDATION] model improved, new model[{mn}]'.format(mn=new_model_name))
        else:
            logger.info(
                '[VALIDATION] model improve failed, model[{mn}]'.format(mn=new_model_name)
            )

if __name__ == '__main__':
    assert(logger)
    logging.basicConfig(filename='validation.log', level=logging.DEBUG)
    validation(base_model_name='resnet_18')
