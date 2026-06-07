import torch
from torchvision.ops import box_convert, nms

from configs.config import S, IMG_SIZE
from configs.training import CONF_THRESHOLD, NMS_IOU_THRESHOLD

new_h, new_w = IMG_SIZE

def decode_pred(pred):
    """
    Decodes predicted output to torchmetrics format

    :param pred:
        Model's activated and reshaped output
        dims [B, S, S, AB, 5 + C]
    :return:
        A list of dictionaries with boxes/scores/labels

        list[
            dict(
              boxes=tensor([[x1, y1, x2, y2]]),
              scores=tensor([conf * Pr(obj)]),
              labels=tensor([t]),
            )
        ]
    """
    conf = pred[..., 4]
    mask = (conf > CONF_THRESHOLD)
    b_idx, gy, gx, a = mask.nonzero(as_tuple=True)

    boxes = pred[b_idx, gy, gx, a, :4]

    # Denormalize, convert cxcy relative to the whole img
    boxes[:, 0] = ((boxes[:, 0] + gx) / S) * new_w
    boxes[:, 1] = ((boxes[:, 1] + gy) / S) * new_h
    boxes[:, 2] = boxes[:, 2] * new_w
    boxes[:, 3] = boxes[:, 3] * new_h

    # Apply sigmoid on classes
    class_probs, labels = torch.sigmoid(pred[b_idx, gy, gx, a, 5:]).max(dim=1)

    # Final detection score (conf * Pr(obj))
    scores = conf[b_idx, gy, gx, a] * class_probs
    boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

    # Just make sure that boxes coordinates falls inside an image,
    # sometimes produces negative values becase converting from cxcywh to xyxy
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, new_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, new_h)

    keep_idx = nms(boxes, scores, NMS_IOU_THRESHOLD)

    # Update batch indexes, after nms b_idx sometimes reduced
    b_idx = b_idx[keep_idx]

    boxes = boxes[keep_idx]
    scores = scores[keep_idx]
    labels = labels[keep_idx]

    return [
        {
            "boxes": boxes[b_idx == i],
            "scores": scores[b_idx == i],
            "labels": labels[b_idx == i]
        } for i in range(pred.shape[0])
    ]

def decode_target(target):
    """
    Decodes target output to torchmetrics format

    :param target:
        Reshaped target output
        dims [B, S, S, AB, 5 + C]
    :return:

        list[
            dict(
              boxes=tensor([[x1, y1, x2, y2]]),
              labels=tensor([t]),
            )
        ]
    """
    b_idx, gy, gx, a = target[..., 4].nonzero(as_tuple=True)

    boxes = target[b_idx, gy, gx, a, :4]
    labels = target[b_idx, gy, gx, a, 5:].argmax(dim=1)

    # Denormalize, convert cxcy relative to the whole img
    boxes[:, 0] = ((boxes[:, 0] + gx) / S) * new_w
    boxes[:, 1] = ((boxes[:, 1] + gy) / S) * new_h
    boxes[:, 2] = boxes[:, 2] * new_w
    boxes[:, 3] = boxes[:, 3] * new_h

    boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

    # Just make sure that boxes coordinates falls inside an image,
    # sometimes produces negative values becase converting from cxcywh to xyxy
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, new_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, new_h)

    return [
        {
         "boxes": boxes[b_idx == i],
         "labels": labels[b_idx == i]
        } for i in range(target.shape[0])
    ]