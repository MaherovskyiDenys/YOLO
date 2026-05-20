import torch

from configs.training import EPOCHS, BACKBONE_LR, BACKBONE_WEIGHT_DECAY
from src.dataset.dataloader import get_loaders
from src.models.model import YOLORes
from src.training.loss import YOLOLoss
from src.training.optimizer import build_optimizer
from src.training.train import run_single_epoch_train
from src.training.validate import run_single_epoch_val

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    model = YOLORes().to(device)
    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    dataset_train, dataset_val = get_loaders()

    loss_func = YOLOLoss()
    optimizer = build_optimizer(model.head)

    for epoch in range(EPOCHS):
        # Unfreeze backbone on n_th epoch
        if epoch == 4:
            for param in model.backbone.parameters():
                param.requires_grad = True

            optimizer.add_param_group(
                {"params": model.backbone.parameters(), "lr": BACKBONE_LR, "weight_decay": BACKBONE_WEIGHT_DECAY}
            )

        train_output = run_single_epoch_train(model, device, dataset_train, loss_func, optimizer)
        val_output = run_single_epoch_val(model, device, dataset_val, loss_func)

        print(f'#{epoch} Train loss: {train_output:.2f} | Val loss: {val_output:.2f}')


if __name__ == "__main__":
    train()