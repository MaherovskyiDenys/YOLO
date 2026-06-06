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

IMG_SIZE = (448, 448)  # (h, w) where h == w

S = int(IMG_SIZE[0] / 32)  # reduced by factor of 32
B = 2
ANCHOR_BOXES = 5
C = len(CLASSES)
