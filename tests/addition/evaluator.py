import torch
from torch.utils.data.dataloader import DataLoader

from tests.constants import NUM_DIGITS, BATCH_SIZE


class Evaluator:
    def __init__(self, train_dataset, test_dataset, model):
        self.powers_of_ten = torch.tensor([[10 ** i for i in range(NUM_DIGITS + 1)][::-1]])
        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
        self.test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        self.model = model

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            qty_correct_train = self._evaluate_dataset(self.train_loader)
            qty_correct_test = self._evaluate_dataset(self.test_loader)
        self.model.train()
        return qty_correct_train, qty_correct_test

    def _evaluate_dataset(self, loader):
        return int(sum([self._qty_correct(inputs) for _, (inputs, _) in enumerate(loader)]))

    def _qty_correct(self, inputs):
        return (self._get_target(inputs) == self._get_prediction(inputs)).sum()

    def _get_target(self, inputs):
        digits_1_int = (inputs[:, :NUM_DIGITS] * self.powers_of_ten[:, 1:]).sum(1)
        digits_2_int = (inputs[:, NUM_DIGITS:NUM_DIGITS * 2] * self.powers_of_ten[:, 1:]).sum(1)
        return digits_1_int + digits_2_int

    def _get_prediction(self, inputs):
        digits_12 = inputs[:, :NUM_DIGITS * 2]
        digits_123 = self.model.generate(digits_12, NUM_DIGITS + 1)
        digits_3 = digits_123[:, -(NUM_DIGITS + 1):]
        digits_3_unflipped = digits_3.flip(1)
        return (digits_3_unflipped * self.powers_of_ten).sum(1)
