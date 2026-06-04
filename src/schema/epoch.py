from src.schema.loss import YOLOLossSchema
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class EpochSchema(YOLOLossSchema):
    mAP: Optional[Dict]
