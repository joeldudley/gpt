import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.dataloader import DataLoader

from trainer.constants import NUM_WORKERS, SAMPLES, BATCH_SIZE, GRAD_NORM_CLIP
from trainer.optimizer import OptimizerFactory


class Trainer:
    def __init__(self, model, train_dataset, batch_end_callback, max_iterations):
        self.batch_end_callback = batch_end_callback
        self.loss = None
        self.iteration = 0
        self.max_iterations = max_iterations

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.optimizer = OptimizerFactory.get(self.model.named_modules(), self.model.named_parameters())

        sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=SAMPLES)
        self.dataloader = DataLoader(train_dataset, sampler=sampler, shuffle=False, pin_memory=True,
                                     batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    def run(self):
        print("Device:", self.device)

        self.model.train()

        data_iter = iter(self.dataloader)
        while True:
            # fetch the next batch and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)
            inputs, targets = [tensor.to(self.device) for tensor in batch]

            # forward the model
            _, self.loss = self.model(inputs, targets)

            # backprop and update the parameters
            self.model.zero_grad(set_to_none=True)
            self.loss.backward()
            clip_grad_norm_(self.model.parameters(), GRAD_NORM_CLIP)
            self.optimizer.step()

            # report back
            self.batch_end_callback(self)

            if self.iteration == self.max_iterations:
                break
            self.iteration += 1
