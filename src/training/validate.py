import torch
from torch import nn
from torch.utils.data import DataLoader


def run_single_epoch_val(model: nn.Module, device: torch.device, dataloader: DataLoader, loss_func: nn.Module) -> float:
    total_cost = 0.0
    batches = 0

    model.eval()
    with torch.no_grad():
        for img, target in dataloader:
            img = img.to(device)
            target = target.to(device)

            output = model(img)

            loss = loss_func(output, target)

            total_cost += loss.item()
            batches += 1

        return total_cost / batches