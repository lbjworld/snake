# coding: utf-8
from __future__ import unicode_literals

import logging
from concurrent import futures
import tushare as ts

from envs.data_loader import data_loader

logger = logging.getLogger(__name__)


def _get_stock_data(code):
    # call env for loading data
    data_loader(code)
    return True


class DataCollector(object):
    def __init__(self, timeout=60):
        self._timeout = timeout

    def get_stock_codes(self, batch_size=1):
        stock_basics = ts.get_stock_basics()
        stock_codes = [
            r[0] + ('.SS' if int(r[0][0]) >= 5 else '.SZ') for r in stock_basics.iterrows()
        ]
        l = len(stock_codes)
        for ndx in range(0, l, batch_size):
            yield stock_codes[ndx:min(ndx + batch_size, l)]

    def run(self, batch_size=10):
        for codes in self.get_stock_codes(batch_size=batch_size):
            logger.info('collecting stocks: {codes}'.format(codes=codes))
            with futures.ProcessPoolExecutor(max_workers=batch_size) as executor:
                _tasks = [executor.submit(_get_stock_data, code) for code in codes]
                try:
                    for future in futures.as_completed(_tasks, timeout=self._timeout):
                        exception = future.exception()
                        if exception:
                            logger.error('collect error: {e}'.format(e=exception))
                            continue
                        logger.info('collect finished')
                except futures.TimeoutError:
                    logger.error('some futures timeout')
            logger.info('finished collect stocks: {codes}'.format(codes=codes))


if __name__ == '__main__':
    assert(logger)
    logging.basicConfig(filename='data_collector.log', level=logging.INFO)
    dc = DataCollector()
    dc.run()
