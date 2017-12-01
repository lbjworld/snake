# coding: utf-8
from __future__ import unicode_literals

import logging
from datetime import date, datetime, timedelta
import tushare as ts

from pipeline.sim_generator import SimGenerator
from pipeline.policy_iterator import PolicyIterator
from pipeline.policy_validator import PolicyValidator

logger = logging.getLogger(__name__)


TRAIN_SIZE = 2000
VALID_SIZE = 200
TEST_SIZE = 300
MAX_GENERATION = 100
EPISODE_LENGTH = 30
SIM_ROUNDS = 100  # total sample size: SIM_ROUNDS * EPISODE_LENGTH
SIM_BATCH_SIZE = 20
SIM_ROUNDS_PER_STEP = 100
IMPROVE_EPOCHS = 100
VALID_ROUNDS = 10
VALID_ROUNDS_PER_STEP = 100

CPU_CORES = 2
TARGET_MIN_VALUE=1.01


def get_ipo_date(code, stock_basics):
    ipo_date = datetime.strptime(str(stock_basics.loc[code]['timeToMarket']), '%Y%m%d').date() \
        if stock_basics.loc[code]['timeToMarket'] else date(2000, 1, 1)
    return ipo_date


def stock_list(min_days):
    now_date = datetime.now().date()
    stock_basics = ts.get_stock_basics()  # get stock basics
    # 过滤掉上市时间小于EPISODE_LENGTH天的
    stock_codes = sorted(
        [
            r[0] + ('.SS' if int(r[0][0]) >= 5 else '.SZ') for r in stock_basics.iterrows()
            if now_date > get_ipo_date(r[0], stock_basics) + timedelta(days=min_days)
        ]
    )
    return stock_codes


def pipeline(base_model_name):
    def gen_model_name(bn, version=0):
        return '{bn}_v{g}'.format(bn=bn, g=version)

    stock_codes = stock_list(min_days=EPISODE_LENGTH)
    assert(len(stock_codes) >= TRAIN_SIZE + VALID_SIZE + TEST_SIZE)
    generation = 0
    model_version = 0
    policy_validator = PolicyValidator(episode_length=EPISODE_LENGTH)
    policy_iter = PolicyIterator(episode_length=EPISODE_LENGTH, target_reward=TARGET_MIN_VALUE)
    current_model_name = gen_model_name(base_model_name, version=model_version)
    # build model and save v0 version
    policy_iter.init_model(model_name=current_model_name)
    while generation < MAX_GENERATION:
        logger.info('start generation: {g}'.format(g=generation))
        # policy evaluation
        sim_gen = SimGenerator(
            train_stocks=stock_codes[:TRAIN_SIZE],
            model_name=current_model_name,
            episode_length=EPISODE_LENGTH,
            explore_rate=1e-01,
            sim_count=SIM_ROUNDS,
            rounds_per_step=SIM_ROUNDS_PER_STEP,
            worker_num=CPU_CORES,
        )
        sim_gen.run(sim_batch_size=SIM_BATCH_SIZE)
        logger.info('policy evaluation finished')
        target_model_name = gen_model_name(base_model_name, version=model_version+1)
        # policy improvement
        policy_iter.improve(
            src=current_model_name,
            target=target_model_name,
            epochs=IMPROVE_EPOCHS,
        )
        logger.info('policy improve finished')
        # policy validation (compare between target and src)
        result_model_name = policy_validator.validate(
            valid_stocks=stock_codes[TRAIN_SIZE:TRAIN_SIZE+VALID_SIZE],
            src=current_model_name,
            target=target_model_name,
            rounds=VALID_ROUNDS,
            rounds_per_step=VALID_ROUNDS_PER_STEP,
            worker_num=CPU_CORES,
        )
        if result_model_name != current_model_name:
            # policy improved !!!
            current_model_name = result_model_name
            model_version += 1
        logger.info('finished generation: {g}\ncurrent model: {mn}'.format(
            g=generation, mn=current_model_name
        ))
        generation += 1


if __name__ == '__main__':
    assert(logger)
    logging.basicConfig(filename='pipeline.log', level=logging.DEBUG)
    pipeline(base_model_name='resnet_18')
