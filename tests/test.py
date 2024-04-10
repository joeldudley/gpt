import random
import unittest

import numpy as np
import torch

from gpt.gpt import GPT
from tests.addition.datasets import get_train_test_datasets
from tests.addition.evaluation import evaluate
from tests.constants import VOCAB_SIZE, NUM_DIGITS, RANDOM_SEED
from train.training import train


class Test(unittest.TestCase):
    def setUp(self):
        self._set_rand_seeds()

        max_seq_len = 3 * NUM_DIGITS + 1 - 1
        self.model = GPT(VOCAB_SIZE, max_seq_len)
        self.train_dataset, self.test_dataset = get_train_test_datasets()

    def test_learns_to_sum_two_digit_numbers(self):
        expected_correct = {0: (79, 9), 500: (462, 22), 1000: (5643, 307), 1500: (9379, 493), 2000: (9500, 500)}

        def callback(iteration):
            self._print_progress(iteration)

            if iteration in expected_correct:
                qty_correct_train, qty_correct_test = evaluate(self.model, self.train_dataset, self.test_dataset)
                expected_correct_train, expected_correct_test = expected_correct[iteration]
                self.assertEqual(expected_correct_train, qty_correct_train)
                self.assertEqual(expected_correct_test, qty_correct_test)

        train(self.model, self.train_dataset, 2000, callback)

    @staticmethod
    def _set_rand_seeds():
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)

    def _print_progress(self, iteration):
        if iteration % 500 == 0:
            qty_correct_train, qty_correct_test = evaluate(self.model, self.train_dataset, self.test_dataset)
            share_correct_train = 100 * qty_correct_train / len(self.train_dataset)
            share_correct_test = 100 * qty_correct_test / len(self.test_dataset)

            print()
            print("Iteration", iteration)
            print("Train score: %.2f%% correct (%d/%d)" % (
                share_correct_train, qty_correct_train, len(self.train_dataset)))
            print("Test score: %.2f%% correct (%d/%d)" % (
                share_correct_test, qty_correct_test, len(self.test_dataset)))

        elif iteration % 10 == 0:
            print('.', end='', flush=True)
