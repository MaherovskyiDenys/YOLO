import torch
from torch import nn
import torch.nn.functional as f
from configs.config import S, B, C
from configs.training import LAMBDA_COORD, LAMBDA_NOOBJ

from torchvision.ops import box_convert, complete_box_iou_loss, box_iou


class YOLOLoss(nn.Module):
    def __init__(self, s=S, b=B, c=C):
        super().__init__()
        self.S = s
        self.B = b
        self.C = c

        self.lambda_coord = LAMBDA_COORD
        self.lambda_noobj = LAMBDA_NOOBJ

        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, predicted, target):
        """
        Calculate loss for
            1. CIoU for responsable boxes
            2. Object Confidence Loss
            3. No-object Confidence loss
            4. Classes loss

        Return:
            loss

        :param predicted: Raw logits
        :param target:
        :return:
        """
        predicted = self._activate(predicted)
        b_idx, gy, gx = target[:, :, :, 4].nonzero(as_tuple=True)

        # Cell Selection
        responsible_cells_target = target[b_idx, gy, gx]
        responsible_cells_predicted = predicted[b_idx, gy, gx]

        # 1. CIoU for responsable boxes

        # Boxes Selection
        true_bboxes = responsible_cells_target[:, :4].clone()
        bboxes_1_predicted = responsible_cells_predicted[:, :4].clone()
        bboxes_2_predicted = responsible_cells_predicted[:, 5:9].clone()

        # Note: Boxes are relative to the cell we need to convert CXCY to global img CXCY
        true_bboxes = self._convert_boxes(true_bboxes, gy, gx)
        bboxes_1_predicted = self._convert_boxes(bboxes_1_predicted, gy, gx)
        bboxes_2_predicted = self._convert_boxes(bboxes_2_predicted, gy, gx)

        # Convert to XYXY format
        true_bboxes_xyxy = box_convert(true_bboxes, in_fmt="cxcywh", out_fmt="xyxy")
        bboxes_1_predicted_xyxy = box_convert(bboxes_1_predicted, in_fmt="cxcywh", out_fmt="xyxy")
        bboxes_2_predicted_xyxy = box_convert(bboxes_2_predicted, in_fmt="cxcywh", out_fmt="xyxy")

        ciou_1 = complete_box_iou_loss(true_bboxes_xyxy, bboxes_1_predicted_xyxy, reduction="none")
        ciou_2 = complete_box_iou_loss(true_bboxes_xyxy, bboxes_2_predicted_xyxy, reduction="none")

        # Choosing responsible box dims: [n] with number of boxes
        responsible_box_mask = ciou_1 < ciou_2

        ciou_loss = torch.where(responsible_box_mask, ciou_1, ciou_2).mean()

        # 2. Object Confidence Loss
        iou_1 = box_iou(true_bboxes_xyxy, bboxes_1_predicted_xyxy, fmt="xyxy").diag()
        iou_2 = box_iou(true_bboxes_xyxy, bboxes_2_predicted_xyxy, fmt="xyxy").diag()

        target_iou = torch.where(responsible_box_mask, iou_1, iou_2).detach()
        confidence_predicted_1 = responsible_cells_predicted[:, 4]
        confidence_predicted_2 = responsible_cells_predicted[:, 9]

        responsible_confidence_predicted = torch.where(responsible_box_mask, confidence_predicted_1, confidence_predicted_2)
        object_confidence_loss = self.bce(responsible_confidence_predicted, target_iou)
        # 3. No-object Confidence loss
        noobject_mask = target[:, :, :, 4] == 0.0

        noobject_confidence_target = target[noobject_mask, 4]
        noobject_confidence_1_predicted = predicted[noobject_mask, 4]
        noobject_confidence_2_predicted = predicted[noobject_mask, 9]

        noobject_loss = self.bce(noobject_confidence_1_predicted, noobject_confidence_target) + self.bce(noobject_confidence_2_predicted, noobject_confidence_target)

        # 4. Classes loss
        classes_target = responsible_cells_target[:, 5 * B:]
        classes_predicted = responsible_cells_predicted[:, 5 * B:]

        cls_loss = self.bce(classes_predicted, classes_target)

        return self.lambda_coord * ciou_loss + object_confidence_loss + self.lambda_noobj * noobject_loss + cls_loss

    def _convert_boxes(self, bboxes, gy, gx):
        """

        :param bboxes: [n, 4]
        :param gy: [y]
        :param gx: [x]
        :return: Relative to whole img
        """

        bboxes[:, 0] = (bboxes[:, 0] + gx) / self.S
        bboxes[:, 1] = (bboxes[:, 1] + gy) / self.S
        return bboxes

    def _activate(self, pred):
        pred = pred.clone()
        box1_xy = torch.sigmoid(pred[:, :, :, :2])
        box1_wh = torch.clamp(f.softplus(pred[:, :, :, 2:4]), min=1e-4, max=1.0)

        box2_xy = torch.sigmoid(pred[:, :, :, 5:7])
        box2_wh = torch.clamp(f.softplus(pred[:, :, :, 7:9]), min=1e-4, max=1.0)

        # Confidence
        box1_conf = pred[:, :, :, 4:5]
        box2_conf = pred[:, :, :, 9:10]

        classes = pred[:, :, :, self.B * 5:]

        return torch.cat([box1_xy, box1_wh, box1_conf, box2_xy, box2_wh, box2_conf, classes], dim=-1)