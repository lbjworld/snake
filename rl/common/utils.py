# coding: utf-8
from __future__ import unicode_literals

import pstats
import StringIO


class Profiling(object):
    def __init__(self, pr):
        self._pr = pr

    def __enter__(self):
        self._pr.enable()
        return self._pr

    def __exit__(self, type, value, traceback):
        self._pr.disable()
        result = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(self._pr, stream=result).sort_stats(sortby)
        ps.print_stats()
        print result.getvalue()
