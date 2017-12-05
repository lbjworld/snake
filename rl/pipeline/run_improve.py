# coding: utf-8
from __future__ import unicode_literals

import os
import logging

from common.filelock import FileLock
from pipeline.policy_iterator import PolicyIterator

logger = logging.getLogger(__name__)


EPISODE_LENGTH = 30
DATA_BUFFER_SIZE = 20000
IMPROVE_STEPS_PER_EPOCH = 100
IMPROVE_BATCH_SIZE = 2048
CURRENT_MODEL_FILE = 'model.current'


def set_current_model(model_name, model_dir='./models'):
    with FileLock(file_name=CURRENT_MODEL_FILE) as lock:
        model_path = os.path.join(model_dir, model_name)
        assert(os.path.exists(model_path))
        with open(lock.file_name, 'w') as f:
            f.write(model_name)
            logger.debug('set current model to [{name}]'.format(name=model_name))


def improvement(base_model_name):
    policy_iter = PolicyIterator(
        episode_length=EPISODE_LENGTH, data_buffer_size=DATA_BUFFER_SIZE
    )
    if not os.path.exists(CURRENT_MODEL_FILE):
        # no current model, init from scratch
        init_model_name = policy_iter.init_model(model_name=base_model_name)
        set_current_model(model_name=init_model_name)
    # policy improvement
    policy_iter.improve(
        model_name=base_model_name,
        steps_per_epoch=IMPROVE_STEPS_PER_EPOCH,
        batch_size=IMPROVE_BATCH_SIZE,
    )


if __name__ == '__main__':
    assert(logger)
    logging.basicConfig(filename='improvement.log', level=logging.DEBUG)
    improvement(base_model_name='resnet_18')
