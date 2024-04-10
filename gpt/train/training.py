from torch.nn.utils import clip_grad_norm_
from torch.utils.data import RandomSampler
from torch.utils.data.dataloader import DataLoader

from gpt.constants import SAMPLES, BATCH_SIZE, GRAD_NORM_CLIP
from gpt.train.optimisation import get_adamw_optimizer


def train(model, train_dataset, iterations, batch_end_callback):
    model = model.to('cpu')
    optimizer = get_adamw_optimizer(model.named_modules(), model.named_parameters())
    sampler = RandomSampler(train_dataset, replacement=True, num_samples=SAMPLES)
    dataloader = DataLoader(train_dataset, sampler=sampler, shuffle=False, pin_memory=True, batch_size=BATCH_SIZE)

    data_iter = iter(dataloader)
    iteration = 0
    while iteration <= iterations:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        inputs, targets = [tensor.to('cpu') for tensor in batch]
        _, loss = model(inputs, targets)

        model.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), GRAD_NORM_CLIP)
        optimizer.step()

        batch_end_callback(iteration)
        iteration += 1
