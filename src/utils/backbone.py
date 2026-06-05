from src.models.model import YOLORes
from torch.optim import Optimizer
from configs.training import BACKBONE_LR, BACKBONE_WEIGHT_DECAY

def freeze_backbone(model: YOLORes) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = False

def unfreeze_backbone(model: YOLORes, optimizer: Optimizer) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = True

    optimizer.add_param_group(
        {"params": model.backbone.parameters(), "lr": BACKBONE_LR, "weight_decay": BACKBONE_WEIGHT_DECAY}
    )