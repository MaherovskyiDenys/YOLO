import json
from pathlib import Path

import torch
from sklearn.cluster import KMeans
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
from torchvision.ops import box_convert, box_iou

from configs.config import S, C, CLASSES, ANCHOR_BOXES


class VOCDatasetYOLO(Dataset):
    def __init__(self, root, year = "2007", image_set = "train", transforms=None):
        self.voc = VOCDetection(
            root=root,
            year=year,
            image_set=image_set,
            download=False
        )

        self.transforms = transforms

        self.classes = {name: i for i, name in enumerate(CLASSES)}

    def __len__(self):
        return len(self.voc)

    def encode_target(self, target):
        root = target["annotation"]

        width = int(root.get("size")["width"])
        height = int(root.get("size")["height"])

        objects = root.get("object")
        # Make sure all the boxes are lists
        if not isinstance(objects, list):
            objects = [objects]

        labels = torch.zeros((S, S, ANCHOR_BOXES, (5 + C)))

        for obj in objects:
            label = obj.get("name")
            difficult = obj.get("difficult")

            # Skip bounding boxes with difficult flag
            if difficult == "1":
                continue

            classes = torch.zeros(C)
            label_idx = self.classes[label]
            classes[label_idx] = 1.0

            bnd = obj["bndbox"]
            xmin, ymin = int(bnd["xmin"]), int(bnd["ymin"])
            xmax, ymax = int(bnd["xmax"]), int(bnd["ymax"])

            # Convert from xyxy to cxcywh
            box = box_convert(torch.tensor([xmin, ymin, xmax, ymax]), in_fmt='xyxy', out_fmt='cxcywh')
            cx, cy, w, h = box.tolist()

            anchors = self.get_anchors()

            # Compare truth box with anchor by measuring highest IoU
            box1 = torch.tensor([0.0, 0.0, w, h]).reshape(1, -1)
            anchor_boxes = torch.zeros(anchors.shape[0], 4)
            anchor_boxes[..., 2:] = anchors

            iou_box = box_iou(box1, anchor_boxes, fmt="cxcywh")

            # Normalize so now values represent % to the image
            cx = cx / width
            cy = cy / height
            w = w / width
            h = h / height

            # Identify in which grid cells is center of the box and clamp
            gy = min(int(cy * S), S - 1)
            gx = min(int(cx * S), S - 1)

            # Identify cx,cy of the object in the grid cell
            cy_cell = cy * S - gy
            cx_cell = cx * S - gx

            idx = iou_box.argmax(1).item()

            # Check if cell and anchor box is already used, by verifying conf
            if labels[gy, gx, idx, 4] > 0.0:
                continue

            labels[gy, gx, idx, :5] = torch.tensor([cx_cell, cy_cell, w, h, 1.0])
            labels[gy, gx, idx, 5:] = classes

        return labels

    def __getitem__(self, idx):
        image, target = self.voc[idx]

        yolo_target = self.encode_target(target)

        if self.transforms:
            # Note: Add feature check for horizontal flip then cx = 1 - cx

            image = self.transforms(image)

        return image, yolo_target

    def _identify_anchors(self):
        """
        Apply K-Means on width and height to identify anchor boxed
        :return:
            Returns anchor boxes as a list with len of ANCHOR_BOXES
        """
        w, h = self._get_wh()

        data = list(zip(w, h))
        kmeans: object|KMeans = KMeans(n_clusters=ANCHOR_BOXES, random_state=0, n_init="auto").fit(data)

        return kmeans.cluster_centers_.astype(int).tolist()

    def _get_wh(self) -> tuple[list, list]:
        """
        Runs over given dataset, outputs widths/heights of all bounding boxes in it

        :return:
            Raw data
            tuple(list[widths], list[heights])
        """
        widths, heights = [], []

        for i, t in self.voc:
            root = t["annotation"]

            objects = root.get("object")

            # Make sure all the boxes are lists
            if not isinstance(objects, list):
                objects = [objects]

            for obj in objects:
                bnd = obj["bndbox"]
                xmin, ymin = int(bnd["xmin"]), int(bnd["ymin"])
                xmax, ymax = int(bnd["xmax"]), int(bnd["ymax"])

                # Convert from xyxy to cxcywh
                box = box_convert(torch.tensor([xmin, ymin, xmax, ymax]), in_fmt='xyxy', out_fmt='cxcywh')
                cx, cy, w, h = box.tolist()

                # Collect width and height of a box
                widths.append(w)
                heights.append(h)

        return widths, heights

    def get_anchors(self) -> Tensor:
        """
        Make sure anchors.json exists and save anchor boxes in it
        :return:
        """
        base = Path(__file__).resolve().parents[2]
        file_path = base / "configs" / "anchors.json"

        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    anchors = json.load(file)
                except json.JSONDecodeError:
                    anchors = None
        else:
            anchors = None

        if not anchors:
            data = {"anchors": self._identify_anchors()}

            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(data, file)

            return torch.tensor(data["anchors"]).reshape(-1, 2)

        return torch.tensor(anchors["anchors"]).reshape(-1, 2)