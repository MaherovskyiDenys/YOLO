from typing import Optional

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from configs.config import ANCHOR_BOXES, C
from src.schema.epoch import EpochSchema
from src.training.decoder import decode_pred, decode_target
from src.training.loss import YOLOLoss
from src.utils.metrics import RunningLoss


def run_epoch(
        model: Module,
        device: torch.device,
        dataloader: DataLoader,
        loss_func: YOLOLoss,
        metric: MeanAveragePrecision,
        optimizer: Optional[Optimizer] = None) -> EpochSchema:
    running_loss = RunningLoss()

    is_train = optimizer is not None

    # Set mode
    model.train() if is_train else model.eval()

    metric.reset()

    gradient = torch.enable_grad() if is_train else torch.no_grad()
    with gradient:
        for img, target in dataloader:
            img = img.to(device)
            target = target.to(device)

            output = model(img)

            B, S, _, _ = output.shape
            output = output.reshape(B, S, S, ANCHOR_BOXES, 5 + C)
            target = target.reshape(B, S, S, ANCHOR_BOXES, 5 + C)

            output = loss_func.activate(output)

            loss_func_output = loss_func(output, target)
            loss = loss_func_output.loss

            metric.update(
                decode_pred(output), decode_target(target)
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss.update(loss_func_output)

        losses = running_loss.compute()

        mAP_results = metric.compute()

        return EpochSchema(**losses, mAP=mAP_results)