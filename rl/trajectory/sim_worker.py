# coding: utf-8
from __future__ import unicode_literals

import time
import pickle
from multiprocessing import Queue, Process

from policy.resnet_trading_model import ResnetTradingModel
from policy.model_policy import ModelTradingPolicy
from trajectory import SimTrajectory


class SimWorker(object):
    @staticmethod
    def sim_worker_func(trajectory, batch_size_per_round, output_queue):
        _start = time.time()
        trajectory.sim_run(batch_size_per_round=batch_size_per_round)
        _end = time.time()
        print('sim worker elapsed: {t}s'.format(t=(_end-_start)*1000.0))
        output_queue.put(trajectory.history)

    @staticmethod
    def collector_func(generation_name, record_num, input_queue):
        data_list = []
        while len(data_list) < record_num:
            batch_data = input_queue.get()
            data_list.extend(batch_data)
        # save to file
        file_name = '{gn}_{t}.pkl'.format(gn=generation_name, t=int(time.time()))
        with open(file_name, 'w') as f:
            pickle.dump(data_list, f)
            print('saved file: {f}'.format(f=file_name))
        print('collector finished')

    def __init__(self, batch_size_per_round=100, max_queue_size=1000):
        self._max_queue_size = max_queue_size
        self._batch_size_per_round = batch_size_per_round

    def _load_state_of_art_policy(self):
        resnet_model = ResnetTradingModel(
            name='test_resnet_model',
            episode_days=self.days,
            feature_num=5,  # open, high, low, close, volume
        )
        model_policy = ModelTradingPolicy(
            action_options=self.env.action_options(),
            model=resnet_model
        )
        return model_policy

    def run(self, generation_name, record_num, process_num=8):
        current_policy = self._load_state_of_art_policy()
        data_queue = Queue(self._max_queue_size)
        sim_workers = []
        for i in range(process_num):
            # TODO: env ?
            trajectory = SimTrajectory(env, current_policy)
            p = Process(SimWorker.sim_worker_func, args=(
                trajectory, self._batch_size_per_round, data_queue,
            ))
            p.start()
            sim_workers.append(p)
        # start collector
        collector = Process(SimWorker.collector_func, args=(
            generation_name, record_num, data_queue,
        ))
        collector.start()
        # wait for collector
        # once collector finished, we got enough records (record_num) and save to file
        collector.join()
        # no more data, close queue
        data_queue.close()
        for p in sim_workers:
            p.join()
