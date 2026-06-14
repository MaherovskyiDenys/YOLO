import torch
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
from torchvision.ops import box_convert, box_iou
from torchvision.tv_tensors import BoundingBoxes

from configs.config import S, C, CLASSES, ANCHOR_BOXES
from src.utils.anchors import get_anchors


class VOCDatasetYOLO(Dataset):
    def __init__(self, root, image_set = "trainval", transforms=None):
        self.voc2007 = VOCDetection(
            root=root,
            year="2007",
            image_set=image_set,
            download=False
        )

        self.voc2012 = None

        if image_set == "trainval":
            self.voc2012 = VOCDetection(
                root=root,
                year="2012",
                image_set=image_set,
                download=False
            )

        # Datasets to parse boxes from
        datasets = [d for d in [self.voc2007, self.voc2012] if d is not None]

        self.anchors = get_anchors(*datasets)

        self.transforms = transforms

        self.classes = {name: i for i, name in enumerate(CLASSES)}

    def __len__(self):
        if self.voc2012 is not None:
            return len(self.voc2007) + len(self.voc2012)

        return len(self.voc2007)

    def parser(self, target):
        """
        Parser will extract bounding boxes and its corresponding labels from an image

        Args:
            target {dict}: Default VOC target format. Represented as dictionary
                - xml example:
                    <annotation>
                        <folder>VOC2007</folder>
                        <filename>000001.jpg</filename>
                        <source>
                            <database>The VOC2007 Database</database>
                            <annotation>PASCAL VOC2007</annotation>
                            <image>flickr</image>
                            <flickrid>341012865</flickrid>
                        </source>
                        <owner>
                            <flickrid>Fried Camels</flickrid>
                            <name>Jinky the Fruit Bat</name>
                        </owner>
                        <size>
                            <width>353</width>
                            <height>500</height>
                            <depth>3</depth>
                        </size>
                        <segmented>0</segmented>
                        <object>
                            <name>dog</name>
                            <pose>Left</pose>
                            <truncated>1</truncated>
                            <difficult>0</difficult>
                            <bndbox>
                                <xmin>48</xmin>
                                <ymin>240</ymin>
                                <xmax>195</xmax>
                                <ymax>371</ymax>
                            </bndbox>
                        </object>
                        <object>
                            <name>person</name>
                            <pose>Left</pose>
                            <truncated>1</truncated>
                            <difficult>0</difficult>
                            <bndbox>
                                <xmin>8</xmin>
                                <ymin>12</ymin>
                                <xmax>352</xmax>
                                <ymax>498</ymax>
                            </bndbox>
                        </object>
                    </annotation>

        Returns:
            tuple(BoundingBoxes, Tensor): Shape ([N, 4], [N,]), Bounding Boxes in xyxy format and labels from an image
            where label are represented as index in configs.config.CLASS variable.
        """
        root = target["annotation"]
        width, height = int(root.get("size")["width"]), int(root.get("size")["height"])

        objects = root.get("object")

        if not isinstance(objects, list):
            objects = [objects]

        boxes = []
        labels = []

        for obj in objects:
            label_idx = self.classes[obj.get("name")]

            bnd = obj["bndbox"]
            xmin, ymin = int(bnd["xmin"]), int(bnd["ymin"])
            xmax, ymax = int(bnd["xmax"]), int(bnd["ymax"])

            boxes.append([xmin, ymin,  xmax, ymax])
            labels.append(label_idx)

        boxes = BoundingBoxes(boxes, format="XYXY", canvas_size=(height, width), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)  # Shape: (N,)

        return boxes, labels

    def encoder(self, boxes: BoundingBoxes, labels: torch.Tensor):
        """
        Encode target into YOLO grid based target

        Assigns each raw bounding box to a specific spatial grid cell (S x S)
        and pairs it with the responsible anchor box determined by the highest IoU

        Args:
            boxes (BoundingBoxes): Shape [N, 4], Bounding boxes
            labels (Tensor): Shape [N,], Indices of labels

        Returns:
            Tensor: Shape [S, S, (5 + C) * ANCHOR_BOXES] for each anchor box in every grid cell representing as
                [cx_cell, cy_cell, width, height, conf, ...classes...(one-hot)]
        """
        height, width = boxes.canvas_size
        grid = torch.zeros((S, S, (5 + C) * ANCHOR_BOXES))

        boxes_cxcywh = box_convert(boxes, in_fmt='xyxy', out_fmt='cxcywh')

        for i in range(boxes.shape[0]):
            cx, cy, w, h = boxes_cxcywh[i].tolist()
            label_idx = labels[i].item()

            # Match Anchor Box by highest IoU
            box = torch.tensor([0.0, 0.0, w, h]).reshape(1, -1)
            anchors = torch.zeros(self.anchors.shape[0], 4)
            anchors[..., 2:] = self.anchors
            iou_box = box_iou(box, anchors, fmt="cxcywh").argmax(1).item()

            # Normalize relative to the image size
            cx = cx / width
            cy = cy / height
            w = w / width
            h = h / height

            # Grid cell assignment
            gy = min(int(cy * S), S - 1)
            gx = min(int(cx * S), S - 1)

            cy_cell = cy * S - gy
            cx_cell = cx * S - gx

            idx = (5 + C) * iou_box

            # If duplicate anchor in cell, skip
            if grid[gy, gx, idx + 4] > 0.0:
                continue

            classes = torch.zeros(C)
            classes[label_idx] = 1.0

            grid[gy, gx, idx:idx + 5] = torch.tensor([cx_cell, cy_cell, w, h, 1.0])
            grid[gy, gx, idx + 5:idx + 5 + C] = classes

        return grid

    def __getitem__(self, idx):
        if self.voc2012 is not None and idx >= len(self.voc2007):
            image, target = self.voc2012[idx - len(self.voc2007)]
        else:
            image, target = self.voc2007[idx]

        boxes, labels = self.parser(target)

        if self.transforms:
            transformed = self.transforms({
                "image": image,
                "boxes": boxes,
                "labels": labels
            })

            image = transformed["image"]
            boxes = transformed["boxes"]
            labels = transformed["labels"]

        yolo_target = self.encoder(boxes, labels)

        return image, yolo_target