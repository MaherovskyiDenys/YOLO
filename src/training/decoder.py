import torch
from torchvision.ops import box_convert, batched_nms

from configs.config import S, IMG_SIZE
from configs.training import CONF_THRESHOLD, NMS_IOU_THRESHOLD, SCORE_THRESHOLD

new_h, new_w = IMG_SIZE

def decode_pred(
        pred,
        conf_threshold: float = CONF_THRESHOLD,
        score_threshold: float = SCORE_THRESHOLD,
        nms_iou_threshold: float = NMS_IOU_THRESHOLD
):
    """
    Decodes predicted model output for torchmetrics

    Args:
        pred (Tensor): Shape [B, S, S, AB, 5 + C], Predicted model output - Activations applied
        score_threshold (float): A minimum threshold for the final detection score (conf * Pr(cls)) to keep the box
        conf_threshold (float): A minimum threshold for the object confidence
        nms_iou_threshold (float): IoU threshold used by NMS to filter out overlapping boxes

    Returns:
        list[dict]: A list of length B (batch size), each element is a dictionary
        representing predicted model output objects formated for torchmetrics.
            "boxes" (Tensor): Shape [N, 4] boxes in abs cords format [x1, y1, x2, y2]
            "scores" (Tensor): Shape [N], final confidence scores (conf * Pr(cls)) for each box
            "labels" (Tensor): Shape [N], corresponding class labels to the boxes
    """
    conf = pred[..., 4]
    mask = (conf > conf_threshold)
    b_idx, gy, gx, a = mask.nonzero(as_tuple=True)

    # Denormalize, convert cxcy relative to the whole img
    cx = ((pred[b_idx, gy, gx, a, 0] + gx) / S) * new_w
    cy = ((pred[b_idx, gy, gx, a, 1] + gy) / S) * new_h
    w = pred[b_idx, gy, gx, a, 2] * new_w
    h = pred[b_idx, gy, gx, a, 3] * new_h

    boxes_cxcywh = torch.stack([cx, cy, w, h], dim=-1)
    # Apply sigmoid on classes
    class_probs, labels = torch.sigmoid(pred[b_idx, gy, gx, a, 5:]).max(dim=1)

    # Final detection score (conf * Pr(obj))
    scores = conf[b_idx, gy, gx, a] * class_probs

    # Remove predictions with low scores
    score_mask = scores > score_threshold

    b_idx = b_idx[score_mask]
    boxes = boxes_cxcywh[score_mask]
    scores = scores[score_mask]
    labels = labels[score_mask]

    boxes_xyxy = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

    # Just make sure that boxes coordinates falls inside an image,
    # sometimes produces negative values becase converting from cxcywh to xyxy
    boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clamp(0, new_w)
    boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clamp(0, new_h)

    results = []

    # Loop through each image in the batch individually
    for i in range(pred.shape[0]):
        # Extract boxes, scores, and labels strictly for image 'i'
        img_mask = (b_idx == i)

        img_boxes = boxes_xyxy[img_mask]
        img_scores = scores[img_mask]
        img_labels = labels[img_mask]

        if img_boxes.numel() == 0:
            # Return empty box if didn't pass thresholds per img
            results.append({
                "boxes": torch.empty((0, 4), device=pred.device),
                "scores": torch.empty((0,), device=pred.device),
                "labels": torch.empty((0,), dtype=torch.long, device=pred.device)
            })
            continue

        keep_idx = batched_nms(img_boxes, img_scores, img_labels, nms_iou_threshold)

        results.append({
            "boxes": img_boxes[keep_idx],
            "scores": img_scores[keep_idx],
            "labels": img_labels[keep_idx]
        })

    return results


def decode_target(target):
    """
    Decodes ground-truth target output for torchmetrics

    Args:
        target (Tensor): Shape [B, S, S, AB, 5 + C], Ground-truth tensor
    Returns:
        list[dict]: A list of length B (batch size), each element is a dictionary
        representing ground-truth objects formated for torchmetrics.
            "boxes" (Tensor): Shape [N, 4] boxes in abs cords format [x1, y1, x2, y2]
            "labels" (Tensor): Shape [N], corresponding class labels to the boxes
    """
    b_idx, gy, gx, a = target[..., 4].nonzero(as_tuple=True)

    labels = target[b_idx, gy, gx, a, 5:].argmax(dim=1)

    # Denormalize, convert cxcy relative to the whole img
    cx = ((target[b_idx, gy, gx, a, 0] + gx) / S) * new_w
    cy = ((target[b_idx, gy, gx, a, 1] + gy) / S) * new_h
    w = target[b_idx, gy, gx, a, 2] * new_w
    h = target[b_idx, gy, gx, a, 3] * new_h

    boxes_cxcywh = torch.stack([cx, cy, w, h], dim=-1)

    boxes_xyxy = box_convert(boxes_cxcywh, in_fmt="cxcywh", out_fmt="xyxy")

    boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clamp(0, new_w)
    boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clamp(0, new_h)

    return [
        {
         "boxes": boxes_xyxy[b_idx == i],
         "labels": labels[b_idx == i]
        } for i in range(target.shape[0])
    ]