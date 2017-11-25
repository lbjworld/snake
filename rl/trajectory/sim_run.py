# coding: utf-8
from __future__ import unicode_literals

import logging
from gym_trading.envs import FastTradingEnv

logger = logging.getLogger(__name__)


def sim_run_func(params):
    from policy.resnet_trading_model import ResnetTradingModel
    from policy.model_policy import ModelTradingPolicy
    from trajectory.sim_trajectory import SimTrajectory

    # get input parameters
    stock_name = params['stock_name']
    episode_length = params['episode_length']
    rounds_per_step = params['rounds_per_step']
    model_name = params['model_name']
    model_dir = params['model_dir']
    model_feature_num = params['model_feature_num']
    sim_explore_rate = params['sim_explore_rate']
    debug = params.get('debug', False)
    # create env
    _env = FastTradingEnv(name=stock_name, days=episode_length)
    _env.reset()
    logger.debug('created env[{name}:{days}]'.format(name=stock_name, days=episode_length))
    # load model
    _model = ResnetTradingModel(
        name=model_name,
        model_dir=model_dir,
        load_model=True,
        episode_days=episode_length,
        feature_num=model_feature_num
    )
    logger.debug('loaded model[{d}/{name}]'.format(d=model_dir, name=model_name))
    _policy = ModelTradingPolicy(action_options=_env.action_options(), model=_model, debug=debug)
    logger.debug('built policy with model[{name}]'.format(name=model_name))
    # start sim trajectory
    _sim = SimTrajectory(
        env=_env, model_policy=_policy, explore_rate=sim_explore_rate, debug=debug
    )
    logger.debug('start simulate trajectory, rounds_per_step({r})'.format(r=rounds_per_step))
    _sim.sim_run(rounds_per_step=rounds_per_step)
    logger.debug('finished simluate trajectory, history size({s})'.format(s=len(_sim.history)))
    # collect data
    return _sim.history
