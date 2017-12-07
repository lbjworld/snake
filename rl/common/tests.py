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
        pool_size = 20
        ds = SimDataSet('./test_dir', pool_size)

        # test load data from scratch
        init_load_size = 10
        mock_get_dir_list.return_value = [
            '{i}.txt'.format(i=init_load_size-i) for i in range(init_load_size)
        ]
        ds._load_single_data_file = mock.MagicMock(return_value=([1], 1))
        ds._load_latest_data()
        self.assertEqual(len(ds._data_pool), init_load_size)
        for idx, item in enumerate(ds._current_file_queue):
            fp, s = item
            self.assertEqual(s, 1)
            self.assertTrue(unicode(init_load_size-idx) in fp)

        # test load additional data
        add_data_size = 20
        total_data_size = init_load_size + add_data_size
        mock_get_dir_list.return_value = [
            '{i}.txt'.format(i=total_data_size-i)
            for i in range(total_data_size)
        ]
        ds._load_single_data_file = mock.MagicMock(return_value=([2], 1))
        ds._load_latest_data()
        self.assertEqual(len(ds._data_pool), pool_size)
        for idx, item in enumerate(ds._current_file_queue):
            fp, s = item
            self.assertEqual(s, 1)
            self.assertTrue(unicode(total_data_size-idx) in fp)

        # check data pool
        for data in ds._data_pool:
            self.assertEqual(data, 2)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
