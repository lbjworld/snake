# coding: utf-8
from __future__ import unicode_literals

import os
import logging

logger = logging.getLogger(__name__)


def get_latest_file(path, name, delimiter='.', order_field_idx=1):
    assert(os.path.exists(path))
    latest_time = 0
    latest_file_name = None
    for root, dirs, files in os.walk(path):
        for file_name in files:
            try:
                t = int(file_name.split(delimiter)[order_field_idx])
            except:
                logger.warn('ignore file: {name}'.format(name=name))
                continue
            if name in file_name and t > latest_time:
                latest_time = t
                latest_file_name = file_name
    return latest_file_name
