CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

BBOX_COLORS = [
    "#FF0000",
    "#00FF00",
    "#0000FF",
    "#FFFF00",
    "#FF00FF",
    "#00FFFF",
    "#FF6600",
    "#FF3399",
    "#9900FF",
    "#008080",
    "#CCFF00",
    "#FF6666",
    "#4B0082",
    "#FFD700",
    "#E6E6FA",
    "#99FFCC",
    "#DC143C",
    "#87CEEB",
    "#7FFF00",
    "#FF8C00"
]

IMG_SIZE = (448, 448)  # (h, w) where h == w

S = int(IMG_SIZE[0] / 32)  # reduced by factor of 32
B = 2
ANCHOR_BOXES = 5
C = len(CLASSES)
