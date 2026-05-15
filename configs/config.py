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

IMG_SIZE = (448, 448)  # (h, w)

S = 7
B = 2
C = 20

# Loss
LAMBDA_COORD = 5
LAMBDA_NOOBJ = 5e-01