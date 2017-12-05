# coding: utf-8
from __future__ import unicode_literals

import logging

from common import settings
from common.dataset import StockDataSet
from pipeline.sim_generator import SimGenerator

logger = logging.getLogger(__name__)


def evaluation(current_model_name):
    ds = StockDataSet()
    stock_codes = ds.stock_list(min_days=settings.EPISODE_LENGTH)
    generation = 0
    while True:
        logger.info('start generation: {g}'.format(g=generation))
        # policy evaluation
        sim_gen = SimGenerator(
            train_stocks=stock_codes[:ds.TRAIN_SIZE],
            model_name=current_model_name,
            episode_length=settings.EPISODE_LENGTH,
            explore_rate=1e-01,
            sim_count=settings.SIM_ROUNDS,
            rounds_per_step=settings.SIM_ROUNDS_PER_STEP,
            worker_num=settings.CPU_CORES,
        )
        sim_gen.run(sim_batch_size=settings.SIM_BATCH_SIZE)
        logger.info('finished generation: {g}\ncurrent model: {mn}'.format(
            g=generation, mn=current_model_name
        ))
        generation += 1


if __name__ == '__main__':
    assert(logger)
    logging.basicConfig(filename='evaluation.log', level=logging.DEBUG)
    evaluation(current_model_name='resnet_18')
