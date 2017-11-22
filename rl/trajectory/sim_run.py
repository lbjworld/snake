# coding: utf-8
from __future__ import unicode_literals

from gym_trading.envs import FastTradingEnv

from policy.resnet_trading_model import ResnetTradingModel
from policy.model_policy import ModelTradingPolicy
from trajectory.sim_trajectory import SimTrajectory


def sim_run_func(params):
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
    # load model
    _model = ResnetTradingModel(
        name=model_name,
        model_dir=model_dir,
        load_model=True,
        episode_days=episode_length,
        feature_num=model_feature_num
    )
    _policy = ModelTradingPolicy(action_options=_env.action_options(), model=_model, debug=debug)
    # start sim trajectory
    _sim = SimTrajectory(
        env=_env, model_policy=_policy, explore_rate=sim_explore_rate, debug=debug
    )
    _sim.run(rounds_per_step=rounds_per_step)
    # collect data
    return _sim.history
