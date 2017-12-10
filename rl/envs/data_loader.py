# coding: utf-8
import os
import logging
from datetime import datetime, date
import pandas as pd
import pandas_datareader.data as web
import tushare as ts

logger = logging.getLogger(__name__)


def cached_filename(name, date):
    return '{name}_{d}_data.pkl.gz'.format(name=name, d=date)


def get_all_hist_data(code, stock_basics, year_interval=3):
    def format_date(d):
        return d.strftime('%Y-%m-%d')

    proxies = ts.get_proxies(count=10)
    now_date = datetime.now().date()
    # 获取上市时间
    ipo_date = datetime.strptime(str(stock_basics.loc[code]['timeToMarket']), '%Y%m%d').date() \
        if stock_basics.loc[code]['timeToMarket'] else date(2000, 1, 1)
    start_date = ipo_date
    end_date = date(start_date.year+year_interval, 1, 1)
    data_frames = []
    while now_date >= start_date:
        try:
            batch_df = ts.get_h_data(
                code, start=format_date(start_date), end=format_date(end_date), proxies=proxies
            )
        except:
            continue
        data_frames.append(batch_df)
        start_date = end_date
        end_date = date(start_date.year+year_interval, 1, 1)
    return pd.concat(data_frames)


def get_data_tushare(code):
    stock_basics = ts.get_stock_basics()  # 获取股票基本情况
    return get_all_hist_data(code, stock_basics)


def data_loader(name, start=None, compression='gzip'):
    if not start:
        start = datetime(2016, 8, 29)
    df = None
    cache_dir = './tmp_data/'
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
        logger.info('tmp data dir: {p} created'.format(p=cache_dir))

    files = os.listdir(cache_dir)
    for f in files:
        if name in f:
            # found cache file
            data_path = os.path.join(cache_dir, f)
            df = pd.read_pickle(data_path, compression=compression)
            break
    if df is None or df.empty:
        # download from web
        df = web.DataReader(name, 'yahoo', start=start, end=datetime.now())
        # df = get_data_tushare(name)
        # cache data
        now_date = datetime.now().date()
        df.to_pickle(
            os.path.join(cache_dir, cached_filename(name, now_date)),
            compression=compression
        )
    return df
