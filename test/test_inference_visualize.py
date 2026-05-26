import torch
from src.models.model import YOLORes
from src.training.loss import YOLOLoss
from src.dataset.dataloader import get_loaders

from torchvision.ops import box_convert, nms
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt

from configs.config import S, IMG_SIZE

CONF_THRESHOLD = 0.5
USE_NMS = True
NMS_IOU_THRESHOLD = 0.5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = YOLORes().to(device)

params = torch.load("../models/test-output.pth", map_location=device)
model.load_state_dict(params)

loss_func = YOLOLoss()

dataset_train, dataset_val = get_loaders(batch_size=1, num_workers=0)

img, target = next(iter(dataset_train))

img = img.to(device)
target = target.to(device)

model.eval()

with torch.no_grad():
    val_output = model(img)
    val_output = loss_func._activate(val_output)

# Box 1

conf_1 = torch.sigmoid(val_output[:, :, :, 4])

mask_1 = conf_1 > CONF_THRESHOLD

b_idx_1, gy_1, gx_1 = mask_1.nonzero(as_tuple=True)

boxes_1 = val_output[b_idx_1, gy_1, gx_1, 0:4]

scores_1 = conf_1[b_idx_1, gy_1, gx_1]

cx_img_1 = (boxes_1[:, 0] + gx_1) / S
cy_img_1 = (boxes_1[:, 1] + gy_1) / S

new_h, new_w = IMG_SIZE

cx_1 = cx_img_1 * new_w
cy_1 = cy_img_1 * new_h

w_1 = boxes_1[:, 2] * new_w
h_1 = boxes_1[:, 3] * new_h

decoded_boxes_1 = torch.empty_like(boxes_1)

decoded_boxes_1[:, 0] = cx_1
decoded_boxes_1[:, 1] = cy_1
decoded_boxes_1[:, 2] = w_1
decoded_boxes_1[:, 3] = h_1

decoded_boxes_1 = box_convert(decoded_boxes_1, in_fmt="cxcywh", out_fmt="xyxy")

# Box 2

conf_2 = torch.sigmoid(val_output[:, :, :, 9])

mask_2 = conf_2 > CONF_THRESHOLD

b_idx_2, gy_2, gx_2 = mask_2.nonzero(as_tuple=True)

boxes_2 = val_output[b_idx_2, gy_2, gx_2, 5:9]

scores_2 = conf_2[b_idx_2, gy_2, gx_2]

cx_img_2 = (boxes_2[:, 0] + gx_2) / S
cy_img_2 = (boxes_2[:, 1] + gy_2) / S

cx_2 = cx_img_2 * new_w
cy_2 = cy_img_2 * new_h

w_2 = boxes_2[:, 2] * new_w
h_2 = boxes_2[:, 3] * new_h

decoded_boxes_2 = torch.empty_like(boxes_2)

decoded_boxes_2[:, 0] = cx_2
decoded_boxes_2[:, 1] = cy_2
decoded_boxes_2[:, 2] = w_2
decoded_boxes_2[:, 3] = h_2

decoded_boxes_2 = box_convert(decoded_boxes_2, in_fmt="cxcywh", out_fmt="xyxy")

# Combine boxes

decoded_boxes = torch.cat([decoded_boxes_1, decoded_boxes_2])

scores = torch.cat([scores_1, scores_2])

decoded_boxes[:, [0, 2]] = decoded_boxes[:, [0, 2]].clamp(0, new_w)
decoded_boxes[:, [1, 3]] = decoded_boxes[:, [1, 3]].clamp(0, new_h)

# Non-maximum suppression

if USE_NMS and len(decoded_boxes) > 0:
    keep_idx = nms(decoded_boxes, scores, NMS_IOU_THRESHOLD)

    decoded_boxes = decoded_boxes[keep_idx]
    scores = scores[keep_idx]

# Denorm img

img = img.squeeze(0)

mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)

std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

img = img * std + mean

img = (img * 255).clamp(0, 255).to(torch.uint8)

# Labels

labels = [f"{score:.2f}" for score in scores]

# Output/Draw

output_img = draw_bounding_boxes(img.cpu(), decoded_boxes.cpu(), labels=labels, width=2)

plt.imshow(output_img.permute(1, 2, 0))
plt.show()