import torch
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection import MeanAveragePrecision

from configs.training import EPOCHS, BACKBONE_LR, BACKBONE_WEIGHT_DECAY, LR, WEIGHT_DECAY
from src.dataset.dataloader import get_loaders
from src.models.model import YOLORes
from src.training.epoch import run_epoch
from src.training.loss import YOLOLoss
from src.training.optimizer import build_optimizer
from src.utils.backbone import freeze_backbone, unfreeze_backbone
from src.utils.checkpoint import Checkpoint
from src.utils.logger import log_epoch, log_config
from src.utils.paths import define_run_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    model = YOLORes().to(device)
    freeze_backbone(model) # Freeze backbone

    dataset_train, dataset_val = get_loaders()

    loss_func = YOLOLoss()
    optimizer = build_optimizer(model.head, lr=LR, weight_decay=WEIGHT_DECAY)
    mAP = MeanAveragePrecision(box_format="xyxy")

    run_name = define_run_name()
    checkpoint = Checkpoint()

    with SummaryWriter(run_name) as writer:
        log_config(writer)

        for epoch in range(EPOCHS):
            # Unfreeze backbone on n_th epoch
            if epoch == 5:
                unfreeze_backbone(model, optimizer)

            train_output = run_epoch(model, device, dataset_train, loss_func, mAP, optimizer)
            val_output = run_epoch(model, device, dataset_val, loss_func, mAP)

            log_epoch(writer, train_output, val_output, epoch)

            checkpoint.update(model, val_output.mAP["map"].item(), run_name)

if __name__ == "__main__":
    train()