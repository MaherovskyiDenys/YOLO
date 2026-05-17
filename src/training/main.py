import torch

from configs.training import EPOCHS
from src.dataset.dataloader import get_loaders
from src.models.model import YOLORes
from src.training.loss import YOLOLoss
from src.training.optimizer import build_optimizer
from src.training.train import run_single_epoch_train
from src.training.validate import run_single_epoch_val

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    model = YOLORes().to(device)

    dataset_train, dataset_val = get_loaders()

    loss_func = YOLOLoss()
    optimizer = build_optimizer(model)

    for epoch in range(EPOCHS):
        train_output = run_single_epoch_train(model, device, dataset_train, loss_func, optimizer)
        val_output = run_single_epoch_val(model, device, dataset_val, loss_func)

        print(f'Train loss: {train_output:.2f} | Val loss: {val_output:.2f}')


if __name__ == "__main__":
    train()