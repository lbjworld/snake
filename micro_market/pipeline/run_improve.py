# coding: utf-8
from __future__ import unicode_literals

import os
import logging

from common import settings
from common.utils import set_current_model
from pipeline.policy_iterator import PolicyIterator

logger = logging.getLogger(__name__)


def improvement(base_model_name):
    policy_iter = PolicyIterator(
        model_dir=settings.MODEL_DATA_DIR,
        data_dir=settings.SIM_DATA_DIR,
        input_shape=(settings.EPISODE_LENGTH, settings.FEATURE_NUM),
        data_buffer_size=settings.DATA_BUFFER_SIZE,
    )
    if not os.path.exists(settings.CURRENT_MODEL_FILE):
        # no current model, init from scratch
        init_model_name = policy_iter.init_model(model_name=base_model_name)
        set_current_model(model_name=init_model_name, model_dir=settings.MODEL_DATA_DIR)
    # policy improvement
    policy_iter.improve(
        model_name=base_model_name,
        steps_per_epoch=settings.IMPROVE_STEPS_PER_EPOCH,
        batch_size=settings.IMPROVE_BATCH_SIZE,
    )


if __name__ == '__main__':
    assert(logger)
    logging.basicConfig(filename='improvement.log', level=logging.DEBUG)
    improvement(base_model_name='resnet_18')
