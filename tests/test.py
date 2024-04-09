import random
import unittest

import numpy as np
import torch

from gpt.gpt import GPT
from tests.addition.evaluator import Evaluator
from tests.addition.samples import Samples
from tests.constants import VOCAB_SIZE, NUM_DIGITS, SEED
from trainer.trainer import Trainer


class Test(unittest.TestCase):
    def setUp(self):
        self._set_rand_seeds()

        max_seq_len = 3 * NUM_DIGITS + 1 - 1
        self.model = GPT(VOCAB_SIZE, max_seq_len)
        self.train_dataset, self.test_dataset = Samples.get_datasets()
        self.evaluator = Evaluator(self.train_dataset, self.test_dataset, self.model)

    def test_learns_to_sum_two_digit_numbers(self):
        expected_correct = {0: (79, 9), 500: (462, 22), 1000: (5643, 307), 1500: (9379, 493), 2000: (9500, 500)}

        def callback(trainer):
            self._print_progress(trainer)

            if trainer.iteration in expected_correct:
                qty_correct_train, qty_correct_test = self.evaluator.evaluate(trainer)
                expected_correct_train, expected_correct_test = expected_correct[trainer.iteration]
                self.assertEqual(expected_correct_train, qty_correct_train)
                self.assertEqual(expected_correct_test, qty_correct_test)

        Trainer(self.model, self.train_dataset, callback, 2000).run()

    @staticmethod
    def _set_rand_seeds():
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    def _print_progress(self, trainer):
        if trainer.iteration % 500 == 0:
            qty_correct_train, qty_correct_test = self.evaluator.evaluate(trainer)
            share_correct_train = 100 * qty_correct_train / len(self.evaluator.train_dataset)
            share_correct_test = 100 * qty_correct_test / len(self.evaluator.test_dataset)

            print()
            print("Iteration", trainer.iteration)
            print("Train score: %.2f%% correct (%d/%d)" % (
                share_correct_train, qty_correct_train, len(self.evaluator.train_dataset)))
            print("Test score: %.2f%% correct (%d/%d)" % (
                share_correct_test, qty_correct_test, len(self.evaluator.test_dataset)))

        elif trainer.iteration % 10 == 0:
            print('.', end='', flush=True)
