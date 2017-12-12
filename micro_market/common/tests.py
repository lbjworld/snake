# coding: utf-8
from __future__ import unicode_literals

import unittest
import mock
import logging

from common.sim_dataset import SimDataSet

logger = logging.getLogger(__name__)


class SimDataSetTestCase(unittest.TestCase):

    @mock.patch('common.sim_dataset.get_dir_list')
    def test_data_rotation(self, mock_get_dir_list):
        pool_size = 10
        ds = SimDataSet('./test_dir', pool_size)

        # test load data from scratch
        exist_files = [
            '{i}.txt'.format(i=i+1) for i in reversed(range(pool_size))
        ]
        mock_get_dir_list.return_value = exist_files
        ds._load_single_data_file = mock.MagicMock(return_value=([1], 1))
        ds._load_latest_data()
        self.assertEqual(len(ds._data_pool), pool_size)
        for idx, item in enumerate(ds._current_file_queue):
            fp, s = item
            self.assertEqual(s, 1)
            self.assertTrue(exist_files[idx] in fp)
        # check data pool
        for data in ds._data_pool:
            self.assertEqual(data, 1)

        # test load one additional data
        exist_files = ['{i}.txt'.format(i=len(exist_files)+1)] + exist_files
        mock_get_dir_list.return_value = exist_files
        ds._load_single_data_file = mock.MagicMock(return_value=([2], 1))
        ds._load_latest_data()
        self.assertEqual(len(ds._data_pool), pool_size)
        for idx, item in enumerate(ds._current_file_queue):
            fp, s = item
            self.assertEqual(s, 1)
            self.assertTrue(exist_files[idx] in fp)
        # check data pool
        for idx, data in enumerate(ds._data_pool):
            if idx != len(ds._data_pool) - 1:
                self.assertEqual(data, 1)
            else:
                self.assertEqual(data, 2)

        # test load many additional data
        exist_files = [
            '{i}.txt'.format(i=len(exist_files)+idx+1) for idx in reversed(range(10))
        ] + exist_files
        mock_get_dir_list.return_value = exist_files
        ds._load_single_data_file = mock.MagicMock(return_value=([3], 1))
        ds._load_latest_data()
        self.assertEqual(len(ds._data_pool), pool_size)
        for idx, item in enumerate(ds._current_file_queue):
            fp, s = item
            self.assertEqual(s, 1)
            self.assertTrue(exist_files[idx] in fp)
        # check data pool
        for data in ds._data_pool:
            self.assertEqual(data, 3)

    @mock.patch('common.sim_dataset.get_dir_list')
    def test_init_load_data(self, mock_get_dir_list):
        pool_size = 10
        ds = SimDataSet('./test_dir', pool_size)
        exist_files = [
            '{i}.txt'.format(i=i+1) for i in reversed(range(20))
        ]
        mock_get_dir_list.return_value = exist_files
        ds._load_single_data_file = mock.MagicMock(return_value=([1], 1))
        ds._load_latest_data()
        self.assertEqual(len(ds._data_pool), pool_size)
        for idx, item in enumerate(ds._current_file_queue):
            fp, s = item
            self.assertEqual(s, 1)
            self.assertTrue(exist_files[idx] in fp)




if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
