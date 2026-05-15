import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from configs.training import EPOCHS
# Data
from src.dataset.dataloader import get_loaders
from src.models.model import YOLORes
from src.training.loss import YOLOLoss
from src.training.optimizer import build_optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        loss_func: nn.Module,
        optimizer: optim.Optimizer | None = None
) -> float:
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    # Metrics
    total_cost = 0.0
    batches = 0

    grad_context = torch.enable_grad() if is_train else torch.no_grad()

    with grad_context:
        for img, target in dataloader:
            img = img.to(device)
            target = target.to(device)

            output = model(img)

            loss = loss_func(output, target)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_cost += loss.item()
            batches += 1

        return total_cost / batches


def train():
    model = YOLORes().to(device)

    dataset_train, dataset_val = get_loaders()

    loss_func = YOLOLoss()
    optimizer = build_optimizer(model)
    
    for epoch in range(EPOCHS):
        train_output = run_epoch(model, dataset_train, loss_func, optimizer)
        val_output = run_epoch(model, dataset_val, loss_func)

        print(f'Train loss: {train_output} | Val loss: {val_output}')

if __name__ == "__main__":
    train()