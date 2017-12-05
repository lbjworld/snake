# coding: utf-8
from __future__ import unicode_literals

import os
import logging

from common.filelock import FileLock

logger = logging.getLogger(__name__)

CURRENT_MODEL_FILE = 'model.current'


def set_current_model(model_name, model_dir='./models'):
    with FileLock(file_name=CURRENT_MODEL_FILE) as lock:
        model_path = os.path.join(model_dir, model_name)
        assert(os.path.exists(model_path))
        with open(lock.file_name, 'w') as f:
            f.write(model_name)
            logger.debug('set current model to [{name}]'.format(name=model_name))
