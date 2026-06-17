import torch
from torchvision.transforms import v2

from configs.config import IMG_SIZE

train_transform = v2.Compose([
    v2.ToImage(),
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), interpolation=v2.InterpolationMode.BILINEAR),
    v2.SanitizeBoundingBoxes(min_area=16.0),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(brightness=0.5, contrast=0.0, saturation=0.5, hue=0.1),
    v2.Resize(IMG_SIZE),

    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(IMG_SIZE),

    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])