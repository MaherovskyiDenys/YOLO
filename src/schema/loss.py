from torch import Tensor
from dataclasses import dataclass

@dataclass
class YOLOLossSchema:
  ciou: Tensor
  obj: Tensor
  noobj: Tensor
  cls: Tensor
  loss: Tensor