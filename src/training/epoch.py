import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from src.schema.epoch import EpochSchema
from src.training.loss import YOLOLoss
from src.training.mAP import decode_pred, decode_target
from torchmetrics.detection import MeanAveragePrecision
from torch.optim import Optimizer
from typing import Optional


def run_epoch(
        model: Module,
        device: torch.device,
        dataloader: DataLoader,
        loss_func: YOLOLoss,
        metric: MeanAveragePrecision,
        optimizer: Optional[Optimizer] = None) -> EpochSchema:
    """
    Supports both datasets uses a switch
    """
    # Metrics
    total_ciou = 0.0
    total_obj = 0.0
    total_noobj = 0.0
    total_cls = 0.0
    total_loss = 0.0

    batches = 0

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

            loss_func_output = loss_func(output, target)

            metric.update(
                decode_pred(output), decode_target(target)
            )

            ciou = loss_func_output.ciou
            obj = loss_func_output.obj
            noobj = loss_func_output.noobj
            cls = loss_func_output.cls
            loss = loss_func_output.loss

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_ciou += ciou.item()
            total_obj += obj.item()
            total_noobj += noobj.item()
            total_cls += cls.item()
            total_loss += loss.item()

            batches += 1

        losses = {
            "ciou": total_ciou / batches,
            "obj": total_obj / batches,
            "noobj": total_noobj / batches,
            "cls": total_cls / batches,
            "loss": total_loss / batches
        }

        # mAP
        map_results = metric.compute()

        return EpochSchema(**losses, mAP=map_results)