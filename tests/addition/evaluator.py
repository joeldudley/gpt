import torch
from torch.utils.data.dataloader import DataLoader

from tests.test_constants.constants import NUM_DIGITS


class Evaluator:
    def __init__(self, train_dataset, test_dataset):
        self.powers_of_ten = torch.tensor([[10 ** i for i in range(NUM_DIGITS + 1)][::-1]])
        self.train_loader = DataLoader(train_dataset, batch_size=100)
        self.test_loader = DataLoader(test_dataset, batch_size=100)

    def evaluate(self, model):
        model.eval()
        with torch.no_grad():
            qty_correct_train = self._evaluate_dataset(model, self.train_loader)
            qty_correct_test = self._evaluate_dataset(model, self.test_loader)
        model.train()
        return qty_correct_train, qty_correct_test

    def _evaluate_dataset(self, model, loader):
        return sum([self._qty_correct(model, inputs) for _, (inputs, _) in enumerate(loader)])

    def _qty_correct(self, model, inputs):
        digits_1_int = (inputs[:, :NUM_DIGITS] * self.powers_of_ten[:, 1:]).sum(1)
        digits_2_int = (inputs[:, NUM_DIGITS:NUM_DIGITS * 2] * self.powers_of_ten[:, 1:]).sum(1)
        target = digits_1_int + digits_2_int

        digits_12 = inputs[:, :NUM_DIGITS * 2]
        digits_123 = model.generate(digits_12, NUM_DIGITS + 1)
        digits_3 = digits_123[:, -(NUM_DIGITS + 1):].flip(1)
        prediction = (digits_3 * self.powers_of_ten).sum(1)

        return (prediction == target).sum()
