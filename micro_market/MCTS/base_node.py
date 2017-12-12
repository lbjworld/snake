# coding: utf-8
from __future__ import unicode_literals

import abc


class BaseNode(object):
    """Base Node for MCTS"""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def step(self, policy):
        """extend one step in tree by using 'policy' """
        raise NotImplemented
