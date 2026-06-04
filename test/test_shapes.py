import torch
from src.models.model import YOLORes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = YOLORes().to(device)

test = torch.randn(2, 3, 448, 448).to(device)
test_2 = torch.randn(2, 3, 640, 640).to(device)
pred = model(test)
print(pred.shape)