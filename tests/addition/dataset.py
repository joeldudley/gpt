from enum import Enum

from torch.utils.data import Dataset

from tests.addition.constants import MAX_TEST_SET_SIZE


class Split(Enum):
    TEST = "Test"
    TRAIN = "Train"


class AdditionDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
