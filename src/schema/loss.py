from dataclasses import dataclass

from torch import Tensor


@dataclass
class YOLOLossSchema:
  """Base schema for YOLO loss components"""
  ciou: Tensor
  obj: Tensor
  noobj: Tensor
  cls: Tensor
  loss: Tensor
