from configs.config import ANCHOR_BOXES, C
import torch

from src.models.model import YOLORes
from src.training.loss import YOLOLoss

def predict(x: torch.Tensor, model: YOLORes, loss_func: YOLOLoss) -> torch.Tensor:
    """Makes a prediction using given input and model, reshapes and activates output"""
    with torch.no_grad():
        output = model(x)

        B, S, _, _ = output.shape
        output = output.reshape(B, S, S, ANCHOR_BOXES, 5 + C)

        predicted = loss_func.activate(output)

        return predicted