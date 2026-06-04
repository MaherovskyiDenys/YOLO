import torch
from torchvision.ops import box_convert, nms

from configs.config import B, S, IMG_SIZE
from configs.training import CONF_THRESHOLD, NMS_IOU_THRESHOLD
from src.training.loss import YOLOLoss

new_h, new_w = IMG_SIZE

def decode_pred(pred):
    """

    :param pred:
        dims [B, 7, 7, 7] raw logits
    :return:
      list[
          dict(
            boxes=tensor([[258.0, 41.0, 606.0, 285.0]]),
            scores=tensor([0.536]),
            labels=tensor([0]),
          )
      ]
    """
    result = []
    loss_func = YOLOLoss()

    with torch.no_grad():
        pred = loss_func._activate(pred)

    # _activate doesn't apply sigmoid on conf
    conf_1 = torch.sigmoid(pred[:, :, :, 4])
    conf_2 = torch.sigmoid(pred[:, :, :, 9])

    mask_1 = conf_1 > CONF_THRESHOLD
    mask_2 = conf_2 > CONF_THRESHOLD

    b_idx_1, gy_1, gx_1 = mask_1.nonzero(as_tuple=True)
    b_idx_2, gy_2, gx_2 = mask_2.nonzero(as_tuple=True)

    for b in range(pred.shape[0]):
        mask_1 = (b_idx_1 == b)
        mask_2 = (b_idx_2 == b)

        img_gy_1 = gy_1[mask_1]
        img_gx_1 = gx_1[mask_1]

        img_gy_2 = gy_2[mask_2]
        img_gx_2 = gx_2[mask_2]

        boxes_1 = pred[b, img_gy_1, img_gx_1, 0:4]
        boxes_2 = pred[b, img_gy_2, img_gx_2, 5:9]

        # cx, cy - relative to the whole image + denormalized
        boxes_1[:, 0] = ((boxes_1[:, 0] + img_gx_1) / S) * new_w
        boxes_1[:, 1] = ((boxes_1[:, 1] + img_gy_1) / S) * new_h
        boxes_2[:, 0] = ((boxes_2[:, 0] + img_gx_2) / S) * new_w
        boxes_2[:, 1] = ((boxes_2[:, 1] + img_gy_2) / S) * new_h

        # w, h - denormalized
        boxes_1[:, 2] = boxes_1[:, 2] * new_w
        boxes_1[:, 3] = boxes_1[:, 3] * new_h
        boxes_2[:, 2] = boxes_2[:, 2] * new_w
        boxes_2[:, 3] = boxes_2[:, 3] * new_h

        scores_1 = conf_1[b, img_gy_1, img_gx_1]
        scores_2 = conf_2[b, img_gy_2, img_gx_2]

        labels_1 = pred[b, img_gy_1, img_gx_1, 5 * B:].argmax(1)
        labels_2 = pred[b, img_gy_2, img_gx_2, 5 * B:].argmax(1)

        boxes = torch.cat([boxes_1, boxes_2], dim=0)
        scores = torch.cat([scores_1, scores_2], dim=0)
        labels = torch.cat([labels_1, labels_2], dim=0)

        boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

        # Just make sure that boxes coordinates falls inside an image,
        # sometimes produces negative values becase converting from cxcywh to xyxy
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, new_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, new_h)

        # Non-Max Suppression
        keep_idx = nms(boxes, scores, NMS_IOU_THRESHOLD)

        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        labels = labels[keep_idx]

        result.append({
            "boxes": boxes,
            "scores": scores,
            "labels": labels
        })

    return result

def decode_target(target):
    """

    :param target:
        dims [B, 7, 7, 7]
    :return:
      list[
          dict(
            boxes=tensor([[258.0, 41.0, 606.0, 285.0]]),
            labels=tensor([0]),
          )
      ]
    """
    b_idx, gy, gx = target[:, :, :, 4].nonzero(as_tuple=True)

    result = []

    for b in range(target.shape[0]):
        mask = (b_idx == b)

        img_gy = gy[mask]
        img_gx = gx[mask]

        boxes = target[b, img_gy, img_gx, 0:4]
        labels = target[b, img_gy, img_gx, 5 * B:].argmax(dim=1)

        # Convert cx, cy
        boxes[:, 0] = ((boxes[:, 0] + img_gx) / S) * new_w
        boxes[:, 1] = ((boxes[:, 1] + img_gy) / S) * new_h
        boxes[:, 2] = boxes[:, 2] * new_w
        boxes[:, 3] = boxes[:, 3] * new_h

        boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

        # Just make sure that boxes coordinates falls inside an image,
        # sometimes produces negative values becase converting from cxcywh to xyxy
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, new_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, new_h)

        result.append({
            "boxes": boxes,
            "labels": labels
        })

    return result