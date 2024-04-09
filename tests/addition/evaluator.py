import torch
from torch.utils.data.dataloader import DataLoader

from tests.constants import NUM_DIGITS


class Evaluator:
    def __init__(self, train_dataset, test_dataset, model):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            qty_correct_train = self._evaluate_dataset(self.train_dataset)
            qty_correct_test = self._evaluate_dataset(self.test_dataset)
        self.model.train()

        return qty_correct_train, qty_correct_test

    def _evaluate_dataset(self, dataset):
        factors = torch.tensor([[10 ** i for i in range(NUM_DIGITS + 1)][::-1]]).to(self.device)
        loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)

        total_correct = 0
        for _, (inputs, _) in enumerate(loader):
            total_correct += self._qty_correct(inputs.to(self.device), factors)

        return total_correct

    def _qty_correct(self, digits, factors):
        digits_12 = digits[:, :NUM_DIGITS * 2]
        # todo - joel - what does this mean?
        # using greedy argmax, not sampling
        digits_123 = self.model.generate(digits_12, NUM_DIGITS + 1, do_sample=False)
        digits_3 = digits_123[:, -(NUM_DIGITS + 1):].flip(1)

        digits_1_int = (digits_12[:, :NUM_DIGITS] * factors[:, 1:]).sum(1)
        digits_2_int = (digits_12[:, NUM_DIGITS:NUM_DIGITS * 2] * factors[:, 1:]).sum(1)
        digits_3_prediction = (digits_3 * factors).sum(1)
        digits_3_target = digits_1_int + digits_2_int

        return int((digits_3_prediction == digits_3_target).cpu().sum())
