# coding: utf-8
from __future__ import unicode_literals

import numpy as np

from base_node import BaseNode


class Edge(object):
    def __init__(self, prior_p, up, down=None):
        assert(prior_p and up)
        # relation data
        self._up_node = up
        self._down_node = down
        # internal data
        self._visit_count = 0
        self._total_reward = 0.0
        self._mean_reward = 0.0
        self._action_probability = prior_p

    def backup(self, v):
        self._total_reward += v
        self._visit_count += 1
        self._mean_reward = self._total_reward / self._visit_count


class TradingNode(BaseNode):
    # global settings
    env = None
    episode_count = 0

    # debug infos
    graph = None  # for drawing graph
    _last_graph_node = None

    @classmethod
    def get_episode_count(cls):
        return cls.episode_count

    @classmethod
    def add_graph_node(cls, name, latest_ticker=None, reward=None):
        if cls.graph:
            if cls._last_graph_node is None:
                cls._last_graph_node = cls.graph
            is_new_node = False
            next_node = filter(lambda x: x.name == name, cls._last_graph_node.children)
            if not next_node:
                cls._last_graph_node = cls._last_graph_node.add_child(name=str(name))
                is_new_node = True
            else:
                assert(len(next_node) == 1)
                cls._last_graph_node = next_node[0]
            if reward is not None:
                cls._last_graph_node.add_features(final_reward=reward)
            if is_new_node:
                if np.any(latest_ticker):
                    # order: open high low close volume
                    cls._last_graph_node.add_features(close=latest_ticker[3])

    @classmethod
    def clean_graph(cls):
        if cls.graph:
            cls._last_graph_node = None

    @classmethod
    def show_graph(cls):
        if cls.graph:
            print(cls.graph.get_ascii(attributes=['name', 'final_reward']))

    def __init__(self, state, up_edge=None):
        self._state = state
        self._up_edge = up_edge
        action_size = len(self.env.action_options())
        self._down_edges = [Edge(prior_p=1.0/action_size, up=self) for i in range(action_size)]

    def _get_klass(self):
        return self.__class__

    @property
    def is_leaf(self):
        return all([e.end for e in self._edges])

    @property
    def is_root(self):
        return bool(not self._up_edge)

    @property
    def q_table(self, t=1.0):
        # do actual play based on current node
        # return pai(action|state)
        _c = [np.power(e._visit_count, 1.0/t) for e in self._down_edges]
        _sum_c = sum(_c)
        return [i/_sum_c for i in _c]

    def set_env(self, env):
        # override class attribute 'env'
        self._get_klass().env = env

    def _select(self, c_puct=1.0):
        # refer to: PUCT algorithm
        total_visit_count = sum([e._visit_count for e in self._down_edges])
        max_v = 0.0
        action = 0
        for a_i, e in enumerate(self._down_edges):
            v = e._mean_reward + c_puct * e._action_probability * np.sqrt(total_visit_count) / (1 + e._visit_count)
            if max_v < v:
                max_v = v
                action = a_i
        return action

    def _backup(self, v):
        current_node = self
        while current_node and not current_node.is_root:
            current_node._up_edge.backup(v)
            current_node = current_node._up_edge._up_node

    def step(self, policy):
        """
            Args:
                policy (Policy): policy object for evaluation
            Returns:
                TradingNode: next node if exist (None if done)
        """
        NodeClass = self._get_klass()
        action = self._select()
        # run in env
        obs, reward, done, info = NodeClass.env.step(action)
        next_edge = self._down_edges[action]
        if not next_edge._down_node:
            # expand new node
            next_node = NodeClass(state=obs, up_edge=next_edge)
            next_edge._down_node = next_node
            # evaluate with policy
            p, v = policy.evaluate(obs)
            # backup
            next_node._backup(v)
        else:
            # reuse exist node
            next_node = next_edge._down_node

        if not done:
            NodeClass.add_graph_node(
                name='{a}'.format(a=action),
                latest_ticker=obs[info['step']]
            )
        else:
            # episode done, reach leaf node
            NodeClass.add_graph_node(
                name='{a}'.format(a=action),
                latest_ticker=obs[info['step']],
                reward=reward
            )
            NodeClass.episode_count += 1
            # clean stats
            NodeClass.clean_graph()
            next_node = None
        return next_node
