# coding: utf-8
from __future__ import unicode_literals


def klass_factory(name, init_args, base_klass):
    klass_attrs = dict()
    for k, v in init_args.items():
        klass_attrs[k] = v
    new_klass = type(str(name), (base_klass, ), klass_attrs)
    return new_klass

