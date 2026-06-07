import json
from pathlib import Path
from typing import Any

base = Path(__file__).resolve().parents[2]
FILE_PATH = base / "configs" / "anchors.json"

def get_anchors() -> dict|None:
    if FILE_PATH.exists():
        with open(FILE_PATH, "r", encoding="utf-8") as file:
            try:
                anchors = json.load(file)
                return anchors
            except json.JSONDecodeError:
                return None
    else:
        return None

def save_anchors(data: dict[str, Any]) -> None:
    with open(FILE_PATH, "w", encoding="utf-8") as file:
        json.dump(data, file)

def get_wh(dataset) -> tuple[list, list]:
    """
    Runs over given VOC 2007 dataset, outputs widths/heights of all bounding boxes

    :return:
        Raw data
        dtype: tuple(list[widths], list[heights])
    """
    widths, heights = [], []

    for _, target in dataset:
        root = target["annotation"]
        objects = root.get("object")

        # Make sure all the boxes are lists
        if not isinstance(objects, list):
            objects = [objects]

        for obj in objects:
            bnd = obj["bndbox"]
            xmin, ymin = int(bnd["xmin"]), int(bnd["ymin"])
            xmax, ymax = int(bnd["xmax"]), int(bnd["ymax"])

            # Convert format from xyxy to cxcywh (wh only)
            w = abs(xmax - xmin)
            h = abs(ymax - ymin)

            # Collect width and height of a box
            widths.append(w)
            heights.append(h)

    return widths, heights