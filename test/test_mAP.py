import torch
from torchmetrics.detection import MeanAveragePrecision

from configs.training import EPOCHS, BACKBONE_LR, BACKBONE_WEIGHT_DECAY, LR, WEIGHT_DECAY
from src.dataset.dataloader import get_loaders
from src.models.model import YOLORes
from src.training.epoch import run_epoch
from src.training.loss import YOLOLoss
from src.training.optimizer import build_optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    model = YOLORes().to(device)
    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    dataset_train, dataset_val = get_loaders()

    loss_func = YOLOLoss()
    mAP = MeanAveragePrecision(box_format="xyxy")
    optimizer = build_optimizer(model.head, lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(EPOCHS):
        if epoch == 5:  # Unfreeze backbone on n_th epoch
            for param in model.backbone.parameters():
                param.requires_grad = True

            optimizer.add_param_group(
                {"params": model.backbone.parameters(), "lr": BACKBONE_LR, "weight_decay": BACKBONE_WEIGHT_DECAY}
            )

        train_output = run_epoch(model, device, dataset_train, loss_func, mAP, optimizer)
        val_output = run_epoch(model, device, dataset_val, loss_func, mAP)

        print("CIoU/Train", train_output.ciou)
        print("Obj/Train", train_output.obj)
        print("Noobj/Train", train_output.noobj)
        print("Cls/Train", train_output.cls)
        print("Loss/Train", train_output.loss)

        print("mAP/Train", train_output.mAP)

        print()

        print("CIoU/Val", val_output.ciou)
        print("Obj/Val", val_output.obj, )
        print("Noobj/Val", val_output.noobj)
        print("Cls/Val", val_output.cls)
        print("Loss/Val", train_output.loss)

        print("mAP/Val", train_output.mAP)



if __name__ == "__main__":
    train()