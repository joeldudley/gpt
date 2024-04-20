from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset
from torch.utils.data import RandomSampler
from torch.utils.data.dataloader import DataLoader

from simplegpt.config.config import MAX_GRAD_NORM, BATCH_SIZE, NUM_SAMPLES
from simplegpt.training.optimisation import get_adamw_optimizer


class Trainer:
    def __init__(self, model: Module, train_dataset: Dataset):
        self.model = model
        self.optimizer = get_adamw_optimizer(model.named_modules(), model.named_parameters())
        self.dataloader = DataLoader(train_dataset, shuffle=False, pin_memory=True, batch_size=BATCH_SIZE,
                                     sampler=RandomSampler(train_dataset, replacement=True, num_samples=NUM_SAMPLES))
        self.data_iter = iter(self.dataloader)

    def train(self, iterations: int, batch_end_callback) -> None:
        for iteration in range(iterations + 1):
            inputs, targets = self._get_batch()
            _, loss = self.model(inputs, targets)

            self.model.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()

            batch_end_callback(iteration)

    def _get_batch(self):
        try:
            return next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            return next(self.data_iter)
