# coding: utf-8
from __future__ import unicode_literals

import numpy as np

from base_node import BaseNode


class TradingNode(BaseNode):
    # global settings
    env = None
    rollout_count = 0

    # debug infos
    graph = None  # for drawing graph
    _last_graph_node = None

    @classmethod
    def get_rollout_count(cls):
        return cls.rollout_count

    @classmethod
    def add_graph_node(cls, name, reward=None):
        if cls.graph:
            if cls._last_graph_node is None:
                cls._last_graph_node = cls.graph
            next_node = filter(lambda x: x.name == name, cls._last_graph_node.children)
            if not next_node:
                cls._last_graph_node = cls._last_graph_node.add_child(name=name)
            else:
                assert(len(next_node) == 1)
                cls._last_graph_node = next_node[0]
            if reward is not None:
                cls._last_graph_node.add_features(final_reward=reward)

    @classmethod
    def clean_graph(cls):
        if cls.graph:
            cls._last_graph_node = None

    @classmethod
    def show_graph(cls):
        if cls.graph:
            print(cls.graph.get_ascii(attributes=['name', 'final_reward']))

    def __init__(self, state, parent_action=None):
        self._state = state
        self._parent_action = parent_action
        self._step_rewards = []
        self._children = [None] * len(self.env.action_options())
        # _final_reward shouldn't be None if it is leaf node
        self._final_reward = None

    def _get_klass(self):
        return self.__class__

    def _save_reward(self, step_reward=None):
        self._step_rewards.append(step_reward)

    @property
    def visit_count(self):
        return len(self._step_rewards)

    @property
    def q_table(self):
        """action -> total_reward"""
        q_table = dict()
        for action in self.env.action_options():
            if self._children[action]:
                rewards = self._children[action].get_recursive_reward()
                q_table[action] = sum(rewards) / float(len(rewards))
        return q_table

    @property
    def average_reward(self):
        if self._step_rewards:
            return sum(self._step_rewards) / float(len(self._step_rewards))
        return 0.0

    @property
    def is_leaf(self):
        return not any(self._children)

    def get_recursive_reward(self):
        """traverse sub-tree to get rewards"""
        # TODO: optimize here, remove recursively function call
        rewards = []
        if self.is_leaf:
            rewards = [self.average_reward]
        else:
            for n in self._children:
                if n:
                    rewards.extend(n.get_recursive_reward())
        return rewards

    def step(self, policy):
        """
            Args:
                policy (Policy): policy object
            Returns:
                TradingNode: next node if exist (None if done)
        """
        assert(self.env and policy)
        NodeClass = self._get_klass()
        if np.any(self._state):
            action = policy.get_action(self._state)
        else:
            action = self.env.action_options()[0]
        # run in env
        obs, reward, done, info = NodeClass.env.step(action)
        next_node = self._children[action]
        if next_node:
            # reuse exist one
            next_node._save_reward(step_reward=reward)
        else:
            # create new node
            next_node = NodeClass(state=obs, parent_action=action)
            next_node._save_reward(step_reward=reward)
            self._children[action] = next_node
        if not done:
            NodeClass.add_graph_node(name='{a}'.format(a=action))
        else:
            # episode done, reach leaf node
            NodeClass.add_graph_node(name='{a}'.format(a=action), reward=reward)
            NodeClass.rollout_count += 1
            # clean stats
            NodeClass.clean_graph()
            next_node = None
        return next_node
