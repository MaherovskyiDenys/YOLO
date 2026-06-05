from torch import nn
from torch.optim import AdamW, Optimizer

from configs.training import LR, WEIGHT_DECAY


def build_optimizer(
    model: nn.Module,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY
) -> Optimizer:
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)