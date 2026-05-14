import torch

from src.dataset.dataloader import get_loaders
import matplotlib.pyplot as plt
from configs.config import S, C, IMG_SIZE, CLASSES, B

from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes

train, val = get_loaders(batch_size=1, num_workers=0)

for i, t in train:
    # print("Image shape", i.shape) # [1, 3, 448, 448]
    # print("Target shape", t.shape) # [1, 7, 7, 30]

    # Remove batch dim
    img = i.squeeze(0)

    b_idx, gy, gx = t[:, :, :, 4].nonzero(as_tuple=True)
    # print("Batch index", b_idx)
    # print("Row", gy)
    # print("Column", gx)
    # print("Not empty lines", t[b_idx, gy, gx])
    boxes = t[b_idx, gy, gx, 0:4]

    # print("Bounding boxes", boxes)
    labels_idx = t[b_idx, gy, gx, 5 * B:].argmax(1)

    labels = []
    for idx in labels_idx:
        labels.append(CLASSES[idx])
    print("labels", labels)

    cx_img = (boxes[:, 0] + gx) / S
    cy_img = (boxes[:, 1] + gy) / S

    new_h, new_w = IMG_SIZE
    cx = cx_img * new_w
    cy = cy_img * new_h
    w = boxes[:, 2] * new_w
    h = boxes[:, 3] * new_h

    decoded_boxes = torch.zeros(boxes.shape)
    decoded_boxes[:, 0] = cx
    decoded_boxes[:, 1] = cy
    decoded_boxes[:, 2] = w
    decoded_boxes[:, 3] = h

    decoded_boxes = box_convert(decoded_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    # print("Decoded bounding boxes", decoded_boxes)
    output_img = draw_bounding_boxes(img, decoded_boxes, width=3)
    # plt.imshow expect (M, N, 3)
    plt.imshow(output_img.permute(1, 2, 0))
    plt.scatter(cx, cy, color='b', s=40)
    plt.grid(visible=True, color='r', linestyle='-', linewidth=0.5)
    plt.yticks([i for i in range(0, IMG_SIZE[0], int(IMG_SIZE[0] / S))])
    plt.xticks([i for i in range(0, IMG_SIZE[1], int(IMG_SIZE[1] / S))])
    plt.show()
