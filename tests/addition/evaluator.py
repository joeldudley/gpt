import torch
from torch.utils.data.dataloader import DataLoader

from tests.addition.dataset import Split


class Evaluator:
    def __init__(self, num_digits, train_dataset, test_dataset, model):
        self.num_digits = num_digits
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model

    def evaluate(self, trainer):
        self.model.eval()
        with torch.no_grad():
            qty_correct_train = self._evaluate_split(trainer, Split.TRAIN)
            qty_correct_test = self._evaluate_split(trainer, Split.TEST)
        self.model.train()

        return qty_correct_train, qty_correct_test

    def _evaluate_split(self, trainer, split):
        dataset = self.train_dataset if split is Split.TRAIN else self.test_dataset
        factors = torch.tensor([[10 ** i for i in range(self.num_digits + 1)][::-1]]).to(trainer.device)
        loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)

        total_correct = 0
        for _, (inputs, _) in enumerate(loader):
            total_correct += self._qty_correct(inputs.to(trainer.device), factors)

        return total_correct

    def _qty_correct(self, digits, factors):
        digits_12 = digits[:, :self.num_digits * 2]
        # todo - joel - what does this mean?
        # using greedy argmax, not sampling
        digits_123 = self.model.generate(digits_12, self.num_digits + 1, do_sample=False)
        digits_3 = digits_123[:, -(self.num_digits + 1):].flip(1)

        digits_1_int = (digits_12[:, :self.num_digits] * factors[:, 1:]).sum(1)
        digits_2_int = (digits_12[:, self.num_digits:self.num_digits * 2] * factors[:, 1:]).sum(1)
        digits_3_prediction = (digits_3 * factors).sum(1)
        digits_3_target = digits_1_int + digits_2_int

        return int((digits_3_prediction == digits_3_target).cpu().sum())
