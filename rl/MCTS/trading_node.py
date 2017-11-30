# coding: utf-8
from __future__ import unicode_literals

import numpy as np
from ete3 import Tree

from base_node import BaseNode


class Edge(object):
    def __init__(self, prior_p, up, down=None):
        assert(up)
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

    @classmethod
    def get_episode_count(cls):
        return cls.episode_count

    def __init__(self, state, up_edge=None, prior_ps=None):
        self._state = state
        self._up_edge = up_edge
        action_size = len(self.env.action_options())
        if prior_ps is None:
            prior_ps = [1.0/action_size for i in range(action_size)]
        self._down_edges = [Edge(prior_p=prior_ps[i], up=self) for i in range(action_size)]

    def _get_klass(self):
        return self.__class__

    # DEBUG helper
    #############################
    def draw_graph(self, node, graph_node):
        for action, down_edge in enumerate(node._down_edges):
            c = graph_node.add_child(name=unicode(action))
            c.add_features(
                N=down_edge._visit_count,
                W=down_edge._total_reward,
                Q=down_edge._mean_reward,
                P=down_edge._action_probability,
            )
            if down_edge._down_node:
                self.draw_graph(down_edge._down_node, c)

    def show_graph(self, name=None):
        t = Tree()
        self.draw_graph(self, t)
        # if name:
        #    t.render(file_name=name)
        # print(t.get_ascii(attributes=['name', 'N', 'W', 'Q', 'P']))
        print(t.get_ascii(attributes=['name', 'N', 'Q']))

    def find_leaf_node(self):
        if self.is_leaf:
            return self
        # find sub edge
        for e in self._down_edges:
            if e._down_node:
                return e._down_node.find_leaf_node()

    def show_final_state(self):
        leaf_node = self.find_leaf_node()
        assert(leaf_node)
        return leaf_node._state
    #############################

    @property
    def is_leaf(self):
        return all([not e._down_node for e in self._down_edges])

    @property
    def is_root(self):
        return bool(not self._up_edge)

    @property
    def q_table(self, t=0.1):
        # do actual play based on current node
        # return pai(action|state)
        _c = [np.power(e._visit_count, 1.0/t) for e in self._down_edges]
        _sum_c = sum(_c)
        return [i/_sum_c for i in _c]

    def set_next_root(self, action):
        next_root = self._down_edges[action]._down_node
        # clean up
        next_root._up_edge = None
        # gc.collect()
        return next_root

    def set_env(self, env):
        # override class attribute 'env'
        self._get_klass().env = env

    def _select(self, c_puct=0.9):
        # refer to: PUCT algorithm
        total_visit_count = sum([e._visit_count for e in self._down_edges])
        vs = []
        for e in self._down_edges:
            v = e._mean_reward + c_puct * e._action_probability * \
                np.sqrt(total_visit_count * 1.0) / (1.0 + e._visit_count)
            vs.append(v)
        if vs[1:] == vs[:-1]:
            return np.random.choice(range(len(vs)))
        return np.argmax(vs)

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
            # evaluate with policy
            p, v = policy.evaluate(obs)
            # expand new node
            next_node = NodeClass(state=obs, up_edge=next_edge, prior_ps=p)
            next_edge._down_node = next_node
            # backup
            next_node._backup(v)
        else:
            # reuse exist node
            next_node = next_edge._down_node

        if done:
            # episode done, reach leaf node
            NodeClass.episode_count += 1
            next_node = None
        return next_node
