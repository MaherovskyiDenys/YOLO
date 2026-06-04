import torch
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection import MeanAveragePrecision

from configs.training import EPOCHS, BACKBONE_LR, BACKBONE_WEIGHT_DECAY, LR, WEIGHT_DECAY
from src.dataset.dataloader import get_loaders
from src.models.model import YOLORes
from src.training.epoch import run_epoch
from src.training.loss import YOLOLoss
from src.training.optimizer import build_optimizer
from src.utils.logger import log_epoch, log_config
from src.utils.paths import define_run_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    model = YOLORes().to(device)
    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    dataset_train, dataset_val = get_loaders()

    loss_func = YOLOLoss()
    optimizer = build_optimizer(model.head, lr=LR, weight_decay=WEIGHT_DECAY)
    mAP = MeanAveragePrecision(box_format="xyxy")

    run_name = define_run_name(f"lr_{LR}_wd_{WEIGHT_DECAY}")

    with SummaryWriter(log_dir=run_name) as writer:
        log_config(writer)

        for epoch in range(EPOCHS):
            if epoch == 5:  # Unfreeze backbone on n_th epoch
                for param in model.backbone.parameters():
                    param.requires_grad = True

                optimizer.add_param_group(
                    {"params": model.backbone.parameters(), "lr": BACKBONE_LR, "weight_decay": BACKBONE_WEIGHT_DECAY}
                )

            train_output = run_epoch(model, device, dataset_train, loss_func, mAP, optimizer)
            val_output = run_epoch(model, device, dataset_val, loss_func, mAP)

            log_epoch(writer, train_output, epoch, "Train")
            log_epoch(writer, val_output, epoch, "Val")

        torch.save(model.state_dict(), f"test-output-{EPOCHS}-{LR}-{BACKBONE_LR}.pth")

if __name__ == "__main__":
    train()