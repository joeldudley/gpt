import torch
from torch.utils.data.dataloader import DataLoader

from tests.test_constants.constants import NUM_DIGITS


def evaluate(model, train_dataset, test_dataset):
    model.eval()
    with torch.no_grad():
        qty_correct_train = _evaluate_dataset(model, train_dataset)
        qty_correct_test = _evaluate_dataset(model, test_dataset)
    model.train()

    return qty_correct_train, qty_correct_test


def _evaluate_dataset(model, dataset):
    factors = torch.tensor([[10 ** i for i in range(NUM_DIGITS + 1)][::-1]])
    loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
    return sum([_qty_correct(model, inputs, factors) for _, (inputs, _) in enumerate(loader)])


def _qty_correct(model, inputs, factors):
    digits_1_int = (inputs[:, :NUM_DIGITS] * factors[:, 1:]).sum(1)
    digits_2_int = (inputs[:, NUM_DIGITS:NUM_DIGITS * 2] * factors[:, 1:]).sum(1)
    target = digits_1_int + digits_2_int

    digits_12 = inputs[:, :NUM_DIGITS * 2]
    digits_123 = model.generate(digits_12, NUM_DIGITS + 1)
    digits_3 = digits_123[:, -(NUM_DIGITS + 1):].flip(1)
    prediction = (digits_3 * factors).sum(1)

    return (prediction == target).sum()
