import torch
from torch import nn
from configs.config import S, B, C

from torchvision.ops import box_convert, complete_box_iou_loss


class YOLOLoss(nn.Module):
    def __init__(self, s=S, b=B, c=C):
        super().__init__()
        self.S = s
        self.B = b
        self.C = c

        self.lambda_coord = 5
        self.lambda_noobj = 0.5

        self.mse = nn.MSELoss()
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

        :param predicted:
        :param target:
        :return:
        """
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
        target_iou = torch.where(responsible_box_mask, 1 - ciou_1, 1 - ciou_2).detach()
        confidence_predicted_1 = responsible_cells_predicted[:, 4]
        confidence_predicted_2 = responsible_cells_predicted[:, 9]

        responsible_confidence_predicted = torch.where(responsible_box_mask, confidence_predicted_1, confidence_predicted_2)
        object_confidence_loss = self.mse(responsible_confidence_predicted, target_iou)

        # 3. No-object Confidence loss
        noobject_mask = target[:, :, :, 4] == 0.0

        noobject_confidence_target = target[noobject_mask, 4]
        noobject_confidence_1_predicted = predicted[noobject_mask, 4]
        noobject_confidence_2_predicted = predicted[noobject_mask, 9]

        noobject_loss = self.mse(noobject_confidence_1_predicted, noobject_confidence_target) + self.mse(noobject_confidence_2_predicted, noobject_confidence_target)

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