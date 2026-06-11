import json
from pathlib import Path

import torch
from sklearn.cluster import KMeans

from configs.config import ANCHOR_BOXES

base = Path(__file__).resolve().parents[2]
FILE_PATH = base / "configs" / "anchors.json"


def get_anchors(*datasets) -> torch.Tensor:
    """
    Gets an anchor boxes from given datasets, saves them locally

    Args:
        *datasets: A list of given datasets

    Returns:
        Tensor: Shape [N, 2], (w,h) of anchor boxes in abs values
    """
    if FILE_PATH.exists():
        try:
            with open(FILE_PATH, "r", encoding="utf-8") as file:
                data = json.load(file)
                if "anchors" in data:
                    return torch.tensor(data["anchors"], dtype=torch.float32)

        except (json.JSONDecodeError, ValueError):
            pass

    if not datasets:
        raise ValueError("anchors.json not found and no datasets provided to generate them")

    calculated_anchors = identify_anchors(*datasets)

    # Save anchors locally
    with open(FILE_PATH, "w", encoding="utf-8") as file:
        json.dump({"anchors": calculated_anchors}, file, indent=4)

    return torch.tensor(calculated_anchors, dtype=torch.float32)


def get_wh(*datasets) -> list[list[int]]:
    """
    Collects width and heights of all the bounding boxes in given datasets

    Args:
        *datasets (list): A list of given datasets

    Returns:
        list[list[int]]: A list of lists with [w, h] of all the bounding boxes in abs values
    """
    boxes = []

    for dataset in datasets:
        if dataset is None:
            continue

        for _, target in dataset:
            objects = target["annotation"].get("object")

            # Make sure all the boxes are lists
            if not isinstance(objects, list):
                objects = [objects]

            for obj in objects:
                bnd = obj["bndbox"]
                # Convert format from xyxy to cxcywh (wh only)
                w = abs(int(bnd["xmax"]) - int(bnd["xmin"]))
                h = abs(int(bnd["ymax"]) - int(bnd["ymin"]))

                # Collect width and height of a box
                boxes.append([w, h])

    return boxes


def identify_anchors(*datasets):
    """
    Applies K-Means on width and height to identify anchor boxes

    Args:
        *datasets: A list of given datasets

    Returns:
        list[list[float]]: A list with anchor boxes [w, h]
    """
    data = get_wh(*datasets)
    kmeans: object | KMeans = KMeans(n_clusters=ANCHOR_BOXES, random_state=0, n_init="auto").fit(data)

    return kmeans.cluster_centers_.astype(float).tolist()

