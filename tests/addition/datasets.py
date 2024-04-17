import itertools
import random

import torch
from torch.utils.data import Dataset

from tests.constants import NUM_DIGITS, TEST_SET_SIZE


class SimpleDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.len = len(self.samples)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.samples[idx]


def get_train_test_datasets():
    all_number_pairs = itertools.combinations_with_replacement(range(10 ** NUM_DIGITS), 2)
    samples = [_get_sample(a, b) for a, b in all_number_pairs]
    random.shuffle(samples)
    return SimpleDataset(samples[TEST_SET_SIZE:]), SimpleDataset(samples[:TEST_SET_SIZE])


# e.g. `9629` becomes 96 + 29 = 125 becomes inputs = `[9, 6, 2, 9, 5, 2]`, targets = `[-1, -1, -1, 5, 2, 1]`
def _get_sample(a: int, b: int):
    a_str = str(a).zfill(NUM_DIGITS)
    b_str = str(b).zfill(NUM_DIGITS)
    target_str = str(a + b).zfill(NUM_DIGITS + 1)[::-1]  # It's easier to learn addition if the sum is flipped.

    samples = [int(s) for s in a_str + b_str + target_str]
    inputs = torch.tensor(samples[:-1])
    targets = torch.tensor(samples[1:])
    targets[:NUM_DIGITS * 2 - 1] = -1  # We mask the inputs for the targets.
    return inputs, targets
