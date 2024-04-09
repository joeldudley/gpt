import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.dataloader import DataLoader

from trainer.constants import NUM_WORKERS, SAMPLES, BATCH_SIZE, GRAD_NORM_CLIP
from trainer.optimizer import OptimizerFactory

# TODO: No need for a class here. Just have a static `train()` method
class Trainer:
    def __init__(self, model, train_dataset):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.optimizer = OptimizerFactory.get(self.model.named_modules(), self.model.named_parameters())

        sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=SAMPLES)
        self.dataloader = DataLoader(train_dataset, sampler=sampler, shuffle=False, pin_memory=True,
                                     batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    def train(self, iterations, batch_end_callback):
        print("Device:", self.device)
        self.model.train()

        data_iter = iter(self.dataloader)
        iteration = 0
        while iteration <= iterations:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)
            inputs, targets = [tensor.to(self.device) for tensor in batch]

            _, loss = self.model(inputs, targets)

            self.model.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), GRAD_NORM_CLIP)
            self.optimizer.step()

            batch_end_callback(self, iteration)
            iteration += 1
