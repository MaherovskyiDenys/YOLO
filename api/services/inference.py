from pathlib import Path

import torch

from configs.config import ANCHORS, CLASSES, IMG_SIZE
from src.dataset.transforms import test_transform
from src.inference.predict import predict
from src.models.model import YOLORes
from src.training.decoder import decode_pred
from src.training.loss import YOLOLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base = Path(__file__).resolve().parents[2]

# Load Model Weights and anchors globally once
model = YOLORes().to(device)
params = torch.load(base / "models/Jun13_15-03-11_LAPTOP-4JD10E9V-mAP.pth", map_location=device)
model.load_state_dict(params["model"])
model.eval()

anchors = torch.tensor(ANCHORS, dtype=torch.float32, device=device)
loss_util = YOLOLoss(anchors).to(device)

def inference_image(img_tensor: torch.Tensor):
    """
    Makes an inference a single image

    Args:
        img_tensor (Tensor): Shape [3, H, W], Tensor of a single image

    Returns:
        dict{
            "boxes" (Tensor) - Shape [N, 4], A tensor with predicted boxes
            "scores" (Tensor) - Shape [N], Confidence sores corresponding to the box
            "labels" (list) - List with labels
        }
    """
    orig_h, orig_w = img_tensor.shape[-2:]

    # Resize to model target size and normalize
    img_normalized = test_transform(img_tensor)
    img_batch = img_normalized.unsqueeze(0)  # Shape: [1, 3, IMG_SIZE, IMG_SIZE]
    img_batch = img_batch.to(device)

    # Model Inference
    pred = predict(img_batch, model, loss_util)
    pred_boxes = decode_pred(pred, conf_threshold=0.5, score_threshold=0.6, nms_iou_threshold=0.5)[0]

    # Scale bounding boxes from model image space back to original coordinates
    if len(pred_boxes["boxes"]) > 0:
        pred_boxes["boxes"][:, [0, 2]] *= (orig_w / IMG_SIZE[1])
        pred_boxes["boxes"][:, [1, 3]] *= (orig_h / IMG_SIZE[0])

    # Convert to lists to match FastAPI format
    pred_boxes["boxes"] = pred_boxes["boxes"].tolist()
    pred_boxes["scores"] = pred_boxes["scores"].tolist()
    pred_boxes["labels"] = [CLASSES[i] for i in pred_boxes["labels"]]

    return pred_boxes