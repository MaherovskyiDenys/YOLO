import torch
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection import MeanAveragePrecision

from configs.config import ANCHOR_BOXES, C
from configs.training import EPOCHS, LR, WEIGHT_DECAY, TEST_BATCH_SIZE, UNFREEZE_BACKBONE_EPOCH
from src.dataset.dataloader import get_loaders
from src.models.model import YOLORes
from src.training.epoch import run_epoch
from src.training.loss import YOLOLoss
from src.training.optimizer import build_optimizer
from src.utils.anchors import get_anchors
from src.utils.backbone import freeze_backbone, unfreeze_backbone
from src.utils.checkpoint import Checkpoint
from src.utils.logger import log_epoch, log_config, log_images
from src.utils.paths import define_run_name
from src.utils.visualization import decode_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    model = YOLORes().to(device)
    freeze_backbone(model)

    train_dataset, _, train_dataloader, test_dataloader = get_loaders()
    anchors = train_dataset.anchors.to(device)

    loss_func = YOLOLoss(anchors).to(device)
    optimizer = build_optimizer(model.head, lr=LR, weight_decay=WEIGHT_DECAY)
    mAP = MeanAveragePrecision(box_format="xyxy")

    run_name = define_run_name()
    checkpoint = Checkpoint()

    with SummaryWriter(run_name) as writer:
        log_config(writer)

        for epoch in range(EPOCHS):
            if epoch == UNFREEZE_BACKBONE_EPOCH:
                unfreeze_backbone(model, optimizer)

            train_output = run_epoch(model, device, train_dataloader, loss_func, mAP, optimizer)
            test_output = run_epoch(model, device, test_dataloader, loss_func, mAP)

            log_epoch(writer, train_output, test_output, epoch)

            if epoch % 1 == 0:
                images, labels = next(iter(test_dataloader))
                images = images.to(device)
                labels = labels.to(device)

                # Inference
                model.eval()
                with torch.no_grad():
                    output = model(images[:TEST_BATCH_SIZE])

                    B, S, _, _ = output.shape
                    output = output.reshape(B, S, S, ANCHOR_BOXES, 5 + C)

                    # Activate logits
                    predicted = loss_func.activate(output)

                # Decode and Log images
                decoded = decode_images(images[:TEST_BATCH_SIZE], labels[:TEST_BATCH_SIZE], predicted)
                log_images(writer, decoded, epoch)

            checkpoint.update(model, optimizer, test_output.mAP["map"].item(), run_name)

if __name__ == "__main__":
    train()