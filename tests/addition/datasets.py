import torch
from torch.utils.data import Dataset

from tests.test_constants.constants import RANDOM_SEED, NUM_DIGITS, TEST_SET_SIZE


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
    all_number_pairs = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(RANDOM_SEED))
    samples = [_get_sample(tensor.item()) for tensor in all_number_pairs]
    return SimpleDataset(samples[TEST_SET_SIZE:]), SimpleDataset(samples[:TEST_SET_SIZE])


# e.g. `9629` becomes 96 + 29 = 125 becomes inputs = `[9, 6, 2, 9, 5, 2]`, targets = `[-1, -1, -1, 5, 2, 1]`
def _get_sample(number_pair: int):
    second_num_max = 10 ** NUM_DIGITS
    first_num = number_pair // second_num_max
    second_num = number_pair % second_num_max
    target = first_num + second_num

    first_num_str = str(first_num).zfill(NUM_DIGITS)
    second_num_str = str(second_num).zfill(NUM_DIGITS)
    target_str = str(target).zfill(NUM_DIGITS + 1)[::-1]  # It's easier to learn addition if the sum is flipped.

    # e.g. `96 + 29 = 125` becomes inputs = [9, 6, 2, 9, 5, 2], targets = [-1, -1, -1, 5, 2, 1]
    samples = [int(s) for s in first_num_str + second_num_str + target_str]
    inputs = torch.tensor(samples[:-1])
    targets = torch.tensor(samples[1:])
    targets[:NUM_DIGITS * 2 - 1] = -1
    return inputs, targets
