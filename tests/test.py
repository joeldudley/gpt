import random
import unittest

import numpy as np
import torch

from addition.addition import Addition
from tests.test_utils import print_progress

SEED = 1337


class Test(unittest.TestCase):
    def setUp(self):
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    def test_learns_to_sum_two_digit_numbers(self):
        num_digits = 2
        addition = Addition(SEED, num_digits)
        expected_correct = {0: (79, 9), 500: (462, 22), 1000: (5643, 307), 1500: (9379, 493), 2000: (9500, 500)}

        def callback(trainer):
            print_progress(addition, trainer)

            if trainer.iteration in expected_correct:
                qty_correct_train, qty_correct_test = addition.evaluate(trainer)
                expected_correct_train, expected_correct_test = expected_correct[trainer.iteration]
                self.assertEqual(expected_correct_train, qty_correct_train)
                self.assertEqual(expected_correct_test, qty_correct_test)

        addition.run(callback, 2000)
