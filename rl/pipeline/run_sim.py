# coding: utf-8
from __future__ import unicode_literals

import logging

from pipeline.sim_generator import SimGenerator
from dataset import StockDataSet

logger = logging.getLogger(__name__)


MAX_GENERATION = 100
EPISODE_LENGTH = 30
SIM_ROUNDS = 1000  # total sample size: SIM_ROUNDS * EPISODE_LENGTH
SIM_BATCH_SIZE = 50
SIM_ROUNDS_PER_STEP = 23

CPU_CORES = 2
TARGET_MIN_VALUE = 1.0


def evaluation(current_model_name):
    ds = StockDataSet()
    stock_codes = ds.stock_list(min_days=EPISODE_LENGTH)
    generation = 0
    while generation < MAX_GENERATION:
        logger.info('start generation: {g}'.format(g=generation))
        # policy evaluation
        sim_gen = SimGenerator(
            train_stocks=stock_codes[:ds.TRAIN_SIZE],
            model_name=current_model_name,
            episode_length=EPISODE_LENGTH,
            explore_rate=1e-01,
            sim_count=SIM_ROUNDS,
            rounds_per_step=SIM_ROUNDS_PER_STEP,
            worker_num=CPU_CORES,
        )
        sim_gen.run(sim_batch_size=SIM_BATCH_SIZE)
        logger.info('finished generation: {g}\ncurrent model: {mn}'.format(
            g=generation, mn=current_model_name
        ))
        generation += 1


if __name__ == '__main__':
    assert(logger)
    logging.basicConfig(filename='evaluation.log', level=logging.DEBUG)
    evaluation(current_model_name='resnet_18')
