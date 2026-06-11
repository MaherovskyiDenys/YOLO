import torch
from torch import Tensor

from configs.config import ANCHOR_BOXES, C, CLASSES, BBOX_COLORS
from src.training.decoder import decode_target, decode_pred
from torchvision.utils import draw_bounding_boxes



def decode_images(
        images: Tensor,
        labels: Tensor,
        predictions: Tensor
):
    """
    Draw ground-truth and predicted bounding boxes on the images

    Args:
        images (Tensor): Shape [B, C, IMG_SIZE, IMG_SIZE], Images
        labels (Tensor): Shape [B, S, S, (5 + C) * AB], Ground-truth labels in a default shape
        predictions (Tensor): Shape [B, S, S, (5 + C) * AB], Predicted model output - Activations applied

    Returns:
        Tensor: Shape [B, 3, IMG_SIZE, IMG_SIZE], Images with ground-truth and predicted bounding boxes, pixels in range 0-1
    """
    device = predictions.device

    images = images.to(device)
    labels = labels.to(device)

    # Reshape labels
    B, S, _, _ = labels.shape
    labels = labels.reshape(B, S, S, ANCHOR_BOXES, 5 + C)
    predictions = predictions.reshape(B, S, S, ANCHOR_BOXES, 5 + C)

    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    images_denorm = images * std + mean

    # Decode targets and predictions
    target_boxes = decode_target(labels)
    pred_boxes = decode_pred(predictions, score_threshold=0.7, nms_iou_threshold=0.5)

    result = torch.empty_like(images_denorm)
    for idx, (img, target, pred) in enumerate(zip(images_denorm, target_boxes, pred_boxes)):
        labels = [label for label in target["labels"]]

        colors_true = []
        labels_true = []
        colors_pred = []
        labels_pred = []

        # Ground-Truth labels
        for label in labels:
            colors_true.append("#32CD32")
            labels_true.append(f"{CLASSES[label]}: {1.0 * 100:.2f}%")

        # Predicted labels
        for score, label in zip(pred["scores"], pred["labels"]):
            colors_pred.append(BBOX_COLORS[label])
            labels_pred.append(f"{CLASSES[label]}: {score * 100:.2f}%")

        result[idx] = draw_bounding_boxes(
            img,
            target["boxes"],
            colors=colors_true,
            labels=labels_true,
            fill_labels=True,
            width=3,
        )

        # Draw predicted bounding boxes over ground-truth
        result[idx] = draw_bounding_boxes(
            result[idx],
            pred["boxes"],
            colors=colors_pred,
            labels=labels_pred,
            fill_labels=True,
            width=3,
        )

    return result
