# coding: utf-8
from __future__ import unicode_literals

import os
import pstats
import StringIO
import logging

from common import settings
from common.filelock import FileLock

logger = logging.getLogger(__name__)


def set_current_model(model_name, model_dir):
    with FileLock(file_name=settings.CURRENT_MODEL_FILE) as lock:
        model_path = os.path.join(model_dir, model_name)
        assert(os.path.exists(model_path))
        with open(lock.file_name, 'w') as f:
            f.write(model_name)
            logger.debug('set current model to [{name}]'.format(name=model_name))


def get_file_name(file_path):
    _, file_name = os.path.split(file_path)
    return file_name.strip()


def get_dir_list(file_dir):
    """list dir, and order by mtime desc"""
    file_paths = [os.path.join(file_dir, f) for f in os.listdir(file_dir)]
    file_paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)  # new -> old
    return file_paths


class Profiling(object):
    def __init__(self, pr):
        self._pr = pr

    def __enter__(self):
        self._pr.enable()
        return self._pr

    def __exit__(self, type, value, traceback):
        self._pr.disable()
        result = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(self._pr, stream=result).sort_stats(sortby)
        ps.print_stats()
        print result.getvalue()
