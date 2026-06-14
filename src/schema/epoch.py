from dataclasses import dataclass
from typing import Dict

from torch import Tensor

from src.schema.loss import YOLOLossSchema


@dataclass
class EpochSchema(YOLOLossSchema):
    """
    Schema used for a consistent output after each epoch

    Inherits all loss fields from YOLOLossSchema and appends
    validation metrics
    """
    mAP: Dict[str, Tensor]
