# coding: utf-8
from __future__ import unicode_literals


def eval_run_func(params):
    from evaluation.evaluator import Evaluator

    # get input parameters
    model_dir = params['model_dir']
    basic_model = params['basic_model']
    evaluate_model = params['evaluate_model']
    input_shape = params['input_shape']
    rounds = params['rounds']
    valid_stocks = params['valid_stocks']
    _evaluator = Evaluator(model_dir=model_dir, input_shape=input_shape)
    BAR, EAR = _evaluator.evaluate(basic_model, evaluate_model, valid_stocks, rounds)
    return BAR, EAR
