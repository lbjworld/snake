# coding: utf-8
from __future__ import unicode_literals

import os
import pickle
import logging
import math
from collections import deque
import numpy as np
from keras.utils import Sequence

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
        self._data_pool = []

    def _load_single_data_file(self, file_path):
        with open(file_path, 'r') as f:
            records = pickle.load(f)
            return records, len(records)

    def _load_new_data(self, file_paths, size):
        _current_size = 0
        for file_path in file_paths:
            r, s = self._load_single_data_file(file_path)
            if _current_size < size:
                self._data_pool.extend(r)
                self._current_file_queue.append((file_path, s))
                _current_size += s
            else:
                break
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
        for i in range(_file_count):
            self._current_file_queue.pop()
        self._data_pool = self._data_pool[_remove_size:]

    def _load_latest_data(self):
        file_paths = os.listdir(self._data_dir)
        if not file_paths:
            raise Exception('no data found in [{d}]'.format(d=self._data_dir))
        file_paths = [os.path.join(self._data_dir, f) for f in file_paths]
        file_paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        if len(self._current_file_queue) == 0:
            # load from scratch
            loaded_size = self._load_new_data(file_paths, self._pool_size)
            logger.debug('load scratch data({s})'.format(s=loaded_size))
        else:
            # load additional files
            assert(len(self._current_file_queue))
            latest_file_path, _ = self._current_file_queue[0]
            if latest_file_path != file_paths[1:]:
                # there are new files added
                loaded_size = self._load_new_data(
                    file_paths[:file_paths.index(latest_file_path)], self._pool_size
                )
                if len(self._data_pool) > self._pool_size:
                    # data pool already full, remove old data
                    self._remove_old_data(loaded_size)
                logger.debug('load incremental data({s})'.format(s=loaded_size))

    def gen_data(self, select_size, shuffle=True):
        data_pool_size = len(self._data_pool)
        if select_size > data_pool_size:
            raise Exception('data pool too small to gen data size({s})'.format(s=select_size))
        select_indices = np.random.choice(range(data_pool_size), select_size)
        if shuffle:
            np.random.shuffle(select_indices)
        _x, p_y, v_y = [None] * select_size, [None] * select_size, [None] * select_size
        for idx, select_idx in enumerate(select_indices):
            r = self._data_pool[select_idx]
            _x[idx] = r['obs']
            p_y[idx] = r['q_table']
            v_y[idx] = r['final_reward']
        return np.array(_x), [np.array(p_y), np.array(v_y)]

    def generator(self, batch_size=2048):
        self._load_latest_data()
        while True:
            yield self.gen_data(select_size=batch_size)
            self._load_latest_data()
