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
        Calculates loss for:
            1. CIoU for responsible anchor boxes
            2. Object Confidence Loss
            3. No-object Confidence loss
            4. Classes loss

        Args:
            predicted (Tensor): Activated Shape [B, S, S, AB, 5 + C] (From run_epoch)
            target (Tensor): Raw Shape [B, S, S, AB, 5 + C]

        Returns:
            YOLOLossSchema: A dataclass containing the mean loss components
        """
        # Identify active mask locations
        b_idx, gy, gx, a = target[..., 4].nonzero(as_tuple=True)

        # Cell Selection
        responsible_cells_target = target[b_idx, gy, gx, a]
        responsible_cells_predicted = predicted[b_idx, gy, gx, a]

        # 1. CIoU Loss
        true_bboxes = responsible_cells_target[..., :4].clone()
        bboxes_predicted = responsible_cells_predicted[..., :4].clone()

        # Convert relative cell CXCY to global img CXCY
        true_bboxes = self._convert_boxes(true_bboxes, gy, gx)
        bboxes_predicted = self._convert_boxes(bboxes_predicted, gy, gx)

        # Convert to XYXY format
        true_xyxy = box_convert(true_bboxes, in_fmt="cxcywh", out_fmt="xyxy")
        bboxes_xyxy = box_convert(bboxes_predicted, in_fmt="cxcywh", out_fmt="xyxy")

        ciou_loss = complete_box_iou_loss(true_xyxy, bboxes_xyxy, reduction="mean")

        # 2. Object Confidence Loss
        # Target confidence should be the IoU score of the predicted bounding box
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

        # Apply scaling multipliers
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
        """Converts local cell bounding boxes to global image coordinates"""
        cx = (bboxes[..., 0] + gx) / self.S
        cy = (bboxes[..., 1] + gy) / self.S
        w = torch.clamp(bboxes[..., 2], min=1e-7)
        h = torch.clamp(bboxes[..., 3], min=1e-7)
        return torch.stack([cx, cy, w, h], dim=-1)

    def activate(self, pred):
        """Applies activation functions on predicted output"""
        output = pred.clone()
        anchors_wh = self.anchors.view(1, 1, 1, ANCHOR_BOXES, 2)  # Change shape to match multiplication

        # Center coordinates: sigmoid maps to cell space [0, 1]
        output[..., 0:2] = torch.sigmoid(pred[..., 0:2])
        # Dimensions: exponential scaling relative to pre-defined anchor sizes
        output[..., 2:4] = anchors_wh * torch.exp(pred[..., 2:4])

        # Confidence score: sigmoid maps to [0, 1]
        output[..., 4:5] = torch.sigmoid(pred[..., 4:5])

        # Class scores: Kept raw for BCEWithLogitsLoss
        output[..., 5:] = pred[..., 5:]

        return output

    @staticmethod
    def _calculate_iou(pred_xyxy, true_xyxy):
        """Calculates exact Intersection over Union metrics"""
        inter_xmin = torch.max(pred_xyxy[:, 0], true_xyxy[:, 0])
        inter_ymin = torch.max(pred_xyxy[:, 1], true_xyxy[:, 1])
        inter_xmax = torch.min(pred_xyxy[:, 2], true_xyxy[:, 2])
        inter_ymax = torch.min(pred_xyxy[:, 3], true_xyxy[:, 3])

        inter_area = torch.clamp(inter_xmax - inter_xmin, min=0) * torch.clamp(inter_ymax - inter_ymin, min=0)

        pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])

        true_area = (true_xyxy[:, 2] - true_xyxy[:, 0]) * (true_xyxy[:, 3] - true_xyxy[:, 1])
        union_area = pred_area + true_area - inter_area

        return inter_area / (union_area + 1e-6)