import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader


def run_single_epoch_train(model: nn.Module, device: torch.device, dataloader: DataLoader, loss_func: nn.Module, optimizer: optim.Optimizer) -> float:
    total_cost = 0.0
    batches = 0

    model.train()
    with torch.enable_grad():
        for img, target in dataloader:
            img = img.to(device)
            target = target.to(device)

            output = model(img)

            loss = loss_func(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_cost += loss.item()
            batches += 1

        return total_cost / batches