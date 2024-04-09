import torch

from tests.constants import MAX_TEST_SET_SIZE, SEED, NUM_DIGITS
from tests.addition.dataset import SimpleDataset


class Samples:
    @staticmethod
    def get_datasets():
        dataset_size = (10 ** NUM_DIGITS) ** 2
        test_set_size = min(int(dataset_size * 0.2), MAX_TEST_SET_SIZE)  # 20% of the whole dataset, max. 500

        samples = Samples._get_all_samples(NUM_DIGITS, dataset_size)
        return SimpleDataset(samples[test_set_size:]), SimpleDataset(samples[:test_set_size])

    @classmethod
    def _get_all_samples(cls, num_digits, dataset_size):
        all_number_pairs = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(SEED))
        return [cls._get_sample(tensor.item(), num_digits) for tensor in all_number_pairs]

    @staticmethod
    def _get_sample(number_pair, num_digits):
        num_width = 10 ** num_digits
        a = number_pair // num_width
        b = number_pair % num_width
        target = a + b

        a_str = str(a).zfill(num_digits)
        b_str = str(b).zfill(num_digits)
        target_str = str(target).zfill(num_digits + 1)[::-1]  # It's easier to learn addition if the sum is flipped

        samples = [int(s) for s in a_str + b_str + target_str]
        inputs = torch.tensor(samples[:-1], dtype=torch.long)
        targets = torch.tensor(samples[1:], dtype=torch.long)
        # todo - joel - what does this mean?
        # we will only train in the output locations. -1 will mask loss to zero
        targets[:num_digits * 2 - 1] = -1

        return inputs, targets
