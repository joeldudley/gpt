import torch


class Samples:
    @classmethod
    def get_all_samples(cls, seed, num_digits, dataset_size):
        all_number_pairs = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(seed))
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
