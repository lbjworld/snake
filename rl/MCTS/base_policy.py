# coding: utf-8
from __future__ import unicode_literals

import abc


class BasePolicy(object):
    """Base Policy"""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_action(self, state):
        """choose action base on current state"""
        raise NotImplemented

    @abc.abstractmethod
    def evaluate(self, state):
        """evaluation state"""
        raise NotImplemented
