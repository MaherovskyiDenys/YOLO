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

IMG_SIZE = (448, 448)  # Format: (h, w), where h == w

S = int(IMG_SIZE[0] / 32)  # Image is reduced by factor of 32
C = len(CLASSES)
ANCHOR_BOXES = 5

ANCHORS = [
    [123.79333020197285, 150.3950211366839],
    [401.73501577287016, 337.9227129337541],
    [46.30997876857653, 58.00212314225159],
    [184.70924690181135, 304.22878932316553],
    [343.81768707482956, 176.95782312925166]
]
