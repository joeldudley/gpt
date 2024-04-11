import torch
from torch.utils.data import Dataset

from tests.test_constants.constants import MAX_TEST_SET_SIZE, RANDOM_SEED, NUM_DIGITS


class SimpleDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.len = len(self.samples)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.samples[idx]


def get_train_test_datasets():
    dataset_size = (10 ** NUM_DIGITS) ** 2
    test_set_size = min(int(dataset_size * 0.2), MAX_TEST_SET_SIZE)  # 20% of the whole dataset, max. 500.

    samples = _get_all_samples(NUM_DIGITS, dataset_size)
    return SimpleDataset(samples[test_set_size:]), SimpleDataset(samples[:test_set_size])


def _get_all_samples(num_digits, dataset_size):
    all_number_pairs = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(RANDOM_SEED))
    return [_get_sample(tensor.item(), num_digits) for tensor in all_number_pairs]


def _get_sample(number_pair, num_digits):
    num_width = 10 ** num_digits
    a = number_pair // num_width
    b = number_pair % num_width
    target = a + b

    a_str = str(a).zfill(num_digits)
    b_str = str(b).zfill(num_digits)
    target_str = str(target).zfill(num_digits + 1)[::-1]  # It's easier to learn addition if the sum is flipped.

    samples = [int(s) for s in a_str + b_str + target_str]
    inputs = torch.tensor(samples[:-1], dtype=torch.long)
    targets = torch.tensor(samples[1:], dtype=torch.long)
    targets[:num_digits * 2 - 1] = -1

    return inputs, targets  # e.g. `96 + 29 = 125` becomes inputs = [9, 6, 2, 9, 5, 2]; targets = [-1, -1, -1, 5, 2, 1]
