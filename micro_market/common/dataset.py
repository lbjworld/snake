# coding: utf-8
from __future__ import unicode_literals

from datetime import date, datetime, timedelta
import tushare as ts


class StockDataSet(object):

    TRAIN_SIZE = 2000
    VALID_SIZE = 200
    TEST_SIZE = 300

    def get_ipo_date(self, code, stock_basics):
        ipo_date = datetime.strptime(str(stock_basics.loc[code]['timeToMarket']), '%Y%m%d').date() \
            if stock_basics.loc[code]['timeToMarket'] else date(2000, 1, 1)
        return ipo_date

    def stock_list(self, min_days):
        now_date = datetime.now().date()
        stock_basics = ts.get_stock_basics()  # get stock basics
        # 过滤掉上市时间小于EPISODE_LENGTH天的
        stock_codes = sorted(
            [
                r[0] + ('.SS' if int(r[0][0]) >= 5 else '.SZ') for r in stock_basics.iterrows()
                if now_date > self.get_ipo_date(r[0], stock_basics) + timedelta(days=min_days)
            ]
        )
        assert(len(stock_codes) >= self.TRAIN_SIZE + self.VALID_SIZE + self.TEST_SIZE)
        return stock_codes

