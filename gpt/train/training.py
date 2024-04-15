from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset
from torch.utils.data import RandomSampler
from torch.utils.data.dataloader import DataLoader

from gpt.constants import NUM_SAMPLES, BATCH_SIZE, MAX_GRAD_NORM
from gpt.train.optimisation import get_adamw_optimizer


def train(model: nn.Module, train_dataset: Dataset, iterations: int, batch_end_callback) -> None:
    optimizer = get_adamw_optimizer(model.named_modules(), model.named_parameters())
    dataloader = DataLoader(train_dataset, shuffle=False, pin_memory=True, batch_size=BATCH_SIZE,
                            sampler=RandomSampler(train_dataset, replacement=True, num_samples=NUM_SAMPLES))

    data_iter = iter(dataloader)
    for iteration in range(iterations + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        inputs, targets = [tensor for tensor in batch]
        _, loss = model(inputs, targets)

        model.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        batch_end_callback(iteration)
