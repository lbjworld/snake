# coding: utf-8
from __future__ import unicode_literals

import pickle
import logging
import math
import threading
from collections import deque
import numpy as np
from keras.utils import Sequence

from common.utils import get_dir_list, get_file_name

logger = logging.getLogger(__name__)


class SimSequence(Sequence):

    def __init__(self, x_set, py_set, vy_set, batch_size):
        self.x, self.py, self.vy = x_set, py_set, vy_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_py = self.py[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_vy = self.vy[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array(batch_x), [np.array(batch_py), np.array(batch_vy)]


class SimDataSet(object):

    def __init__(self, data_dir, pool_size):
        self._data_dir = data_dir
        self._pool_size = pool_size
        self._current_file_queue = deque()  # new -> old
        self._data_pool = []  # old -> new
        self._lock = threading.Lock()

    def _load_single_data_file(self, file_path):
        with open(file_path, 'r') as f:
            records = pickle.load(f)
            return records, len(records)

    def _load_new_data(self, file_paths, size):
        _current_size = 0
        _current_data_blks = []
        _current_files = [f for f, s in self._current_file_queue]
        for file_path in file_paths:
            # order: from new to old
            if file_path in _current_files:
                # ignore already processed file
                continue
            r, s = self._load_single_data_file(file_path)
            if _current_size < size:
                _current_data_blks.append((file_path, r, s))
                _current_size += s
            else:
                break
        for blk in reversed(_current_data_blks):
            # from old to new
            self._data_pool.extend(blk[1])
            self._current_file_queue.appendleft((blk[0], blk[2]))
            logger.debug('add new data file: {fs}'.format(fs=blk[0]))
        return _current_size

    def _remove_old_data(self, size):
        assert(size)
        _remove_size = 0
        _file_count = 0
        for file_path, data_size in reversed(self._current_file_queue):
            if _remove_size + data_size > size:
                break
            _remove_size += data_size
            _file_count += 1
        _remove_files = []
        for i in range(_file_count):
            f, s = self._current_file_queue.pop()
            _remove_files.append(f)
        logger.debug('remove data files: {fs}'.format(fs=_remove_files))
        self._data_pool = self._data_pool[_remove_size:]
        logger.debug('remove old data, files({fc}), size({s})'.format(
            fc=_file_count, s=_remove_size)
        )
        return _remove_size

    def _load_latest_data(self):
        self._lock.acquire()
        file_paths = get_dir_list(self._data_dir)
        if not file_paths:
            raise Exception('no data found in [{d}]'.format(d=self._data_dir))
        if len(self._current_file_queue) == 0:
            # load from scratch
            loaded_size = self._load_new_data(file_paths, self._pool_size)
            logger.debug('load scratch data({s})'.format(s=loaded_size))
        else:
            # load additional files
            assert(len(self._current_file_queue))
            latest_file_path, _ = self._current_file_queue[0]
            latest_file_name = get_file_name(latest_file_path)
            if latest_file_name != get_file_name(file_paths[0]):
                # there are new files added
                latest_idx = 0
                for i, f in enumerate(file_paths):
                    if latest_file_name in f:
                        latest_idx = i
                        break
                assert(latest_idx)
                loaded_size = self._load_new_data(
                    file_paths[:latest_idx], self._pool_size
                )
                logger.debug('load incremental data({s})'.format(s=loaded_size))
                current_pool_size = len(self._data_pool)
                if current_pool_size > self._pool_size:
                    # data pool already full, remove old data
                    removed_size = self._remove_old_data(current_pool_size - self._pool_size)
                    logger.debug('remove old data({s})'.format(s=removed_size))
        self._lock.release()

    def gen_data(self, select_size, shuffle=True):
        data_pool_size = len(self._data_pool)
        if select_size > data_pool_size:
            raise Exception('data pool too small to gen data size({s})'.format(s=select_size))
        logger.info('current data pool size({s})'.format(s=data_pool_size))
        select_indices = np.random.choice(range(data_pool_size), select_size)
        if shuffle:
            np.random.shuffle(select_indices)
        _x, p_y, v_y = [None] * select_size, [None] * select_size, [None] * select_size
        for idx, select_idx in enumerate(select_indices):
            r = self._data_pool[select_idx]
            _x[idx] = r['obs']
            p_y[idx] = r['q_table']
            v_y[idx] = r['final_reward']
        return np.expand_dims(np.array(_x), axis=3), [np.array(p_y), np.array(v_y)]

    def generator(self, batch_size=2048):
        self._load_latest_data()
        while True:
            yield self.gen_data(select_size=batch_size)
            self._load_latest_data()
