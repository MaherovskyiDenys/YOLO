import torch
import torch.nn as nn

from configs.config import C, B

from torchvision.models import resnet18, ResNet18_Weights

class YOLORes(nn.Module):
    def __init__(self, boxes: int = B, classes: int = C):
        super().__init__()
        self.b, self.c = boxes, classes

        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*(list(resnet.children())[:-2]))

        self.head = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            # Fully Convolutional Head
            nn.Conv2d(1024, self.b * 5 + self.c, kernel_size=1)
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass

        :param x: Tensor [B, C, H, W]
        :return: Tensor permuted to [B, H, W, C]
        """
        features = self.backbone(x)
        output = self.head(features).permute(0, 2, 3, 1)

        return output