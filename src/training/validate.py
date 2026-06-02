import torch
from torch import nn
from torch.utils.data import DataLoader

from src.schema.epoch import EpochSchema
from src.training.loss import YOLOLoss


def run_single_epoch_val(model: nn.Module, device: torch.device, dataloader: DataLoader, loss_func: YOLOLoss) -> EpochSchema:
    total_ciou = 0.0
    total_obj = 0.0
    total_noobj = 0.0
    total_cls = 0.0
    total_cost = 0.0
    batches = 0

    model.eval()
    with torch.no_grad():
        for img, target in dataloader:
            img = img.to(device)
            target = target.to(device)

            output = model(img)

            loss_func_output = loss_func(output, target)

            ciou = loss_func_output.ciou
            obj = loss_func_output.obj
            noobj = loss_func_output.noobj
            cls = loss_func_output.cls
            loss = loss_func_output.loss

            total_ciou += ciou.item()
            total_obj += obj.item()
            total_noobj += noobj.item()
            total_cls += cls.item()
            total_cost += loss.item()

            batches += 1

        losses = {
            "ciou": total_ciou / batches,
            "obj": total_obj / batches,
            "noobj": total_noobj / batches,
            "cls": total_cls / batches,
            "loss": total_cost / batches
        }

        return EpochSchema(**losses)