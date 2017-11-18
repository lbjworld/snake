# coding: utf-8
from __future__ import unicode_literals

import gc
import numpy as np
from tqdm import tqdm

from utils import klass_factory
from trading_node import TradingNode


class MCTSBuilder(object):
    def __init__(self, gym_env, debug=False):
        assert(gym_env)
        self._debug = debug
        self._gym_env = gym_env
        self._root_node = None

    @property
    def node_klass(self):
        copy_env = self._gym_env
        klass_init_args = {
            'env': copy_env,
        }
        if self._debug:
            from ete3 import Tree
            klass_init_args.update({
                'graph': Tree()
            })
        return klass_factory(
            'Env_{name}_TradingNode'.format(name=copy_env.name),
            init_args=klass_init_args,
            base_klass=TradingNode
        )

    def clean_up(self):
        # clean up
        self._root_node = None
        gc.collect()

    def _get_policy(self, policies, probability_dist=None):
        if probability_dist:
            assert(len(policies) == len(probability_dist))
        return np.random.choice(policies, p=probability_dist)

    def run_once(self, policies, probability_dist=None, env_snapshot=None):
        if not self._root_node:
            # init node
            self._root_node = self.node_klass(state=None)
        # episode start
        assert(self._root_node)
        if env_snapshot:
            # recover gym env from env_snapshot if exist
            self._gym_env.recover(env_snapshot)
        else:
            # simply reset env
            self._gym_env.reset()
        current_node = self._root_node
        while current_node:
            policy = self._get_policy(policies, probability_dist)
            current_node = current_node.step(policy)
        # episode end
        return self._root_node

    def run_batch(self, policies, probability_dist=None, episode=100, env_snapshot=None):
        for idx in tqdm(range(episode)):
            self.run_once(
                policies=policies,
                probability_dist=probability_dist,
                env_snapshot=env_snapshot
            )
        return self._root_node
