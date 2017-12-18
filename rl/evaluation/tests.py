# coding: utf-8
from __future__ import unicode_literals

import unittest

from common import settings
from policy.resnet_trading_model import ResnetTradingModel
from evaluation.evaluator import Evaluator


class EvaluatorTestCase(unittest.TestCase):
    def setUp(self):
        self.model_dir = './test_dir'
        self.valid_stocks = ['000333.SZ', '600016.SS']
        self.input_shape = (settings.EPISODE_LENGTH, settings.FEATURE_NUM)
        # build model and save to model_dir
        b_model = ResnetTradingModel(
            name='basic',
            model_dir=self.model_dir,
            load_model=False,
            input_shape=self.input_shape,
        )
        self.basic_model = b_model.save_model(self.model_dir, 'basic')
        e_model = ResnetTradingModel(
            name='evaluate',
            model_dir=self.model_dir,
            load_model=False,
            input_shape=self.input_shape,
        )
        self.evaluate_model = e_model.save_model(self.model_dir, 'evaluate')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.model_dir)

    def test_evaluate(self):
        evaluator = Evaluator(model_dir=self.model_dir, input_shape=self.input_shape)
        BAR, EAR = evaluator.evaluate(
            basic_model=self.basic_model,
            evaluate_model=self.evaluate_model,
            valid_stocks=self.valid_stocks,
            rounds=3
        )
        print BAR, EAR
        self.assertNotEqual(BAR, EAR)


if __name__ == '__main__':
    unittest.main()
