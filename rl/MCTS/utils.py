# coding: utf-8
from __future__ import unicode_literals


def klass_factory(name, init_args, base_klass):
    klass_attrs = dict()
    for k, v in init_args.items():
        klass_attrs[k] = v
    new_klass = type(str(name), (base_klass, ), klass_attrs)
    return new_klass


def fast_moving(env, policy, steps=1):
    assert(env and policy)
    state = None
    for i in range(steps):
        action = policy.get_action(state)
        obs, _, done, _ = env.step(action)
        if done:
            break
        state = obs
