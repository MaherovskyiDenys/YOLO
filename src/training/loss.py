import torch
from torch import nn
from torchvision.ops import box_convert, complete_box_iou_loss

from configs.config import S, C, ANCHOR_BOXES, IMG_SIZE
from configs.training import LAMBDA_COORD, LAMBDA_NOOBJ
from src.schema.loss import YOLOLossSchema


class YOLOLoss(nn.Module):
    def __init__(self, anchors: torch.Tensor):
        super().__init__()

        self.anchors = anchors / IMG_SIZE[0]  # Normalize given anchor boxes

        self.S = S
        self.C = C

        self.lambda_coord = LAMBDA_COORD
        self.lambda_noobj = LAMBDA_NOOBJ

        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(self, predicted, target) -> YOLOLossSchema:
        """
        Calculates loss for
            1. CIoU for responsable anchor boxes
            2. Object Confidence Loss
            3. No-object Confidence loss
            4. Classes loss

        :param predicted:
            Tensor dims [B, S, S, AB, (5 + C)] - Already activated logits
        :param target:
            Tensor dims [B, S, S, AB, (5 + C)]
        :return:
            Dataclass YOLOLossSchema with losses (ciou, obj, noobj, cls, loss)
        """
        b_idx, gy, gx, a = target[..., 4].nonzero(as_tuple=True)

        # Cell Selection
        responsible_cells_target = target[b_idx, gy, gx, a]
        responsible_cells_predicted = predicted[b_idx, gy, gx, a]

        # 1. CIoU
        true_bboxes = responsible_cells_target[..., :4].clone()
        bboxes_predicted = responsible_cells_predicted[..., :4].clone()

        # convert CXCY to global img CXCY
        true_bboxes = self._convert_boxes(true_bboxes, gy, gx)
        bboxes_predicted = self._convert_boxes(bboxes_predicted, gy, gx)

        # Convert to XYXY format
        true_xyxy = box_convert(true_bboxes, in_fmt="cxcywh", out_fmt="xyxy")
        bboxes_xyxy = box_convert(bboxes_predicted, in_fmt="cxcywh", out_fmt="xyxy")

        ciou_loss = complete_box_iou_loss(true_xyxy, bboxes_xyxy, reduction="mean")
        # 2. Object Confidence Loss
        iou_targets = self._calculate_iou(bboxes_xyxy.detach(), true_xyxy)

        conf_pred = responsible_cells_predicted[..., 4]
        obj_conf_loss = self.mse(conf_pred, iou_targets)

        # 3. No-object Confidence loss
        noobj_mask = target[..., 4] == 0.0

        noobj_conf_target = target[noobj_mask, 4]
        noobj_conf_pred = predicted[noobj_mask, 4]

        noobject_loss = self.mse(noobj_conf_pred, noobj_conf_target)
        # 4. Classes loss
        classes_target = responsible_cells_target[..., 5:]
        classes_predicted = responsible_cells_predicted[..., 5:]

        cls_loss = self.bce(classes_predicted, classes_target)

        # Return
        ciou = self.lambda_coord * ciou_loss
        noobject = self.lambda_noobj * noobject_loss

        losses = {
            "ciou": ciou,
            "obj": obj_conf_loss,
            "noobj": noobject,
            "cls": cls_loss,
            "loss": ciou + obj_conf_loss + noobject + cls_loss
        }

        return YOLOLossSchema(**losses)

    def _convert_boxes(self, bboxes, gy, gx):
        """
        Converts bounding boxes' relation to the whole img

        :param bboxes: Tensor dims [N, 4]
        :param gy: Tensor dims [n]
        :param gx: Tensor dims [n]
        :return:
            Tensor dims [N, 4] - Bounding boxes that are relative to the whole img
        """

        cx = (bboxes[..., 0] + gx) / self.S
        cy = (bboxes[..., 1] + gy) / self.S
        w = bboxes[..., 2]
        h = bboxes[..., 3]
        return torch.stack([cx, cy, w, h], dim=-1)

    def activate(self, pred):
        """
        Applies activation functions on predicted output

        :param pred:
            Tensor dims [B, S, S, AB, 5 + C]
        :return:
            Tensor dims [B, S, S, AB, 5 + C] - Activation applied
        """

        output = torch.zeros_like(pred)
        anchors_wh = self.anchors.view(1, 1, 1, ANCHOR_BOXES, 2) # change shape to match multiplication

        output[..., 0:2] = torch.sigmoid(pred[..., 0:2])
        output[..., 2:4] = anchors_wh * torch.exp(pred[..., 2:4])

        output[..., 4:5] = torch.sigmoid(pred[..., 4:5])

        output[..., 5:] = pred[..., 5:]

        return output

    @staticmethod
    def _calculate_iou(pred_xyxy, true_xyxy):

        inter_xmin = torch.max(pred_xyxy[:, 0], true_xyxy[:, 0])
        inter_ymin = torch.max(pred_xyxy[:, 1], true_xyxy[:, 1])
        inter_xmax = torch.min(pred_xyxy[:, 2], true_xyxy[:, 2])
        inter_ymax = torch.min(pred_xyxy[:, 3], true_xyxy[:, 3])

        inter_area = torch.clamp(inter_xmax - inter_xmin, min=0) * torch.clamp(inter_ymax - inter_ymin, min=0)

        pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])

        true_area = (true_xyxy[:, 2] - true_xyxy[:, 0]) * (true_xyxy[:, 3] - true_xyxy[:, 1])
        union_area = pred_area + true_area - inter_area

        return inter_area / (union_area + 1e-6)