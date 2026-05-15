import torch
from torchvision.transforms import v2

from configs.config import IMG_SIZE

train_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(IMG_SIZE),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(IMG_SIZE),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])