# coding: utf-8
from __future__ import unicode_literals

from base_node import BaseNode


class TradingNode(BaseNode):
    env = None
    rollout_count = 0

    @classmethod
    def get_rollout_count(cls):
        return cls.rollout_count

    def __init__(self, state, parent_action=None):
        self._state = state
        self._parent_action = parent_action
        self._step_rewards = []
        self._children = [None] * len(self.env.action_options())

    def _get_klass(self):
        return self.__class__

    def _save_reward(self, step_reward=None):
        self._step_rewards.append(step_reward)

    @property
    def visit_count(self):
        return len(self._step_rewards)

    def step(self, policy):
        """
            Args:
                policy (Policy): policy object
            Returns:
                TradingNode: next node if exist (None if done)
        """
        assert(self.env and policy)
        NodeClass = self._get_klass()
        action = policy.get_action(self._state)
        # run in env
        obs, reward, done, info = NodeClass.env.step(action)
        next_node = None
        if not done:
            # episode haven't done
            next_node = self._children[action]
            if next_node:
                # reuse exist one
                next_node._save_reward(step_reward=reward)
            else:
                # create new node
                assert(obs.get('ticker').shape)
                next_node = NodeClass(state=obs['ticker'], parent_action=action)
                next_node._save_reward(step_reward=reward)
                self._children[action] = next_node
        else:
            # episode done, reach leaf node
            NodeClass.rollout_count += 1
        return next_node
