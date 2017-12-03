# coding: utf-8
from __future__ import unicode_literals

import os
import logging

from common.dataset import StockDataSet
from common.filelock import FileLock
from pipeline.policy_iterator import PolicyIterator
from pipeline.policy_validator import PolicyValidator

logger = logging.getLogger(__name__)


MAX_GENERATION = 100
EPISODE_LENGTH = 30
IMPROVE_EPOCHS = 100
VALID_ROUNDS = 100
VALID_ROUNDS_PER_STEP = 23

CPU_CORES = 2
TARGET_MIN_VALUE = 1.0
CURRENT_MODEL_FILE = 'model.current'


def set_current_model(model_name, model_dir='./models'):
    with FileLock(file_name=CURRENT_MODEL_FILE) as lock:
        model_path = os.path.join(model_dir, model_name)
        assert(os.path.exists(model_path))
        with open(lock.file_name, 'w') as f:
            f.write(model_name)
            logger.debug('set current model to [{name}]'.format(name=model_name))


def improvement(base_model_name):
    def gen_model_name(bn, version=0):
        return '{bn}_v{g}'.format(bn=bn, g=version)

    ds = StockDataSet()
    stock_codes = ds.stock_list(min_days=EPISODE_LENGTH)
    generation = 0
    policy_validator = PolicyValidator(episode_length=EPISODE_LENGTH)
    policy_iter = PolicyIterator(episode_length=EPISODE_LENGTH, target_reward=TARGET_MIN_VALUE)
    init_model_name = policy_iter.init_model(model_name=base_model_name)
    set_current_model(model_name=init_model_name)
    while True:
        logger.info('start generation: {g}'.format(g=generation))
        # policy improvement
        target_model_name = policy_iter.improve(
            model_name=base_model_name, epochs=IMPROVE_EPOCHS)
        logger.info('policy improve finished')
        # policy validation (compare between target and src)
        improved = policy_validator.validate(
            valid_stocks=stock_codes[ds.TRAIN_SIZE:ds.TRAIN_SIZE+ds.VALID_SIZE],
            base=base_model_name,
            target=target_model_name,
            rounds=VALID_ROUNDS,
            rounds_per_step=VALID_ROUNDS_PER_STEP,
            worker_num=CPU_CORES,
        )
        if improved:
            set_current_model(model_name=target_model_name)
        logger.info('finished generation: {g}'.format(g=generation))
        generation += 1


if __name__ == '__main__':
    assert(logger)
    logging.basicConfig(filename='improvement.log', level=logging.DEBUG)
    improvement(base_model_name='resnet_18')
