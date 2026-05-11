from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs.training import BATCH_SIZE, NUM_WORKERS
from src.dataset.dataset import VOCDatasetYOLO
from src.dataset.transforms import train_transform, val_transform

base = Path(__file__).resolve().parents[2]


def get_loaders(
    root: str = "data/raw",
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = torch.cuda.is_available(),
):
    train_dataset = VOCDatasetYOLO(base / root, year="2007", image_set="train", transforms=train_transform)
    val_dataset = VOCDatasetYOLO(base / root, year="2007", image_set="val", transforms=val_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_dataloader, val_dataloader