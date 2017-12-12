# coding: utf-8
from __future__ import unicode_literals

import gc

from utils import klass_factory
from trading_node import TradingNode


class MCTSBuilder(object):
    def __init__(self, gym_env, init_node=None, debug=False):
        assert(gym_env)
        self._debug = debug
        self._gym_env = gym_env
        self._root_node = init_node

    @property
    def node_klass(self):
        copy_env = self._gym_env
        klass_init_args = {
            'env': copy_env,
        }
        return klass_factory(
            'Env_{name}_TradingNode'.format(name=copy_env.name),
            init_args=klass_init_args,
            base_klass=TradingNode
        )

    def clean_up(self):
        # clean up
        self._root_node = None
        gc.collect()

    def run_once(self, policy, env_snapshot=None):
        if not self._root_node:
            # init node
            self._root_node = self.node_klass(state=None)
        # episode start
        if env_snapshot:
            # recover gym env from env_snapshot if exist
            self._gym_env.recover(env_snapshot)
        else:
            # simply reset env
            self._gym_env.reset()
        # recover node's env
        self._root_node.set_env(self._gym_env)
        current_node = self._root_node
        while current_node:
            current_node = current_node.step(policy)
        # episode end
        return self._root_node

    def run_batch(self, policy, batch_size=100, env_snapshot=None):
        idx_list = range(batch_size)
        if self._debug:
            from tqdm import tqdm
            idx_list = tqdm(idx_list)
        for idx in idx_list:
            self.run_once(
                policy=policy,
                env_snapshot=env_snapshot
            )
        return self._root_node
