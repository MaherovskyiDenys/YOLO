import torch
from torch.utils.tensorboard import SummaryWriter

from configs.training import EPOCHS, BACKBONE_LR, BACKBONE_WEIGHT_DECAY, LR, WEIGHT_DECAY
from src.dataset.dataloader import get_loaders
from src.models.model import YOLORes
from src.training.loss import YOLOLoss
from src.training.optimizer import build_optimizer
from src.training.train import run_single_epoch_train
from src.training.validate import run_single_epoch_val
from src.utils.paths import define_run_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    model = YOLORes().to(device)
    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    dataset_train, dataset_val = get_loaders()

    loss_func = YOLOLoss()
    optimizer = build_optimizer(model.head, lr=LR, weight_decay=WEIGHT_DECAY)

    run_name = define_run_name(f"lr_{LR}_wd_{WEIGHT_DECAY}")

    with SummaryWriter(log_dir=run_name) as writer:
        writer.add_text("hparams", f"lr:{LR}\nweight_decay:{WEIGHT_DECAY}")

        for epoch in range(EPOCHS):
            if epoch == 10:  # Unfreeze backbone on n_th epoch
                for param in model.backbone.parameters():
                    param.requires_grad = True

                optimizer.add_param_group(
                    {"params": model.backbone.parameters(), "lr": BACKBONE_LR, "weight_decay": BACKBONE_WEIGHT_DECAY}
                )

            train_output = run_single_epoch_train(model, device, dataset_train, loss_func, optimizer)
            val_output = run_single_epoch_val(model, device, dataset_val, loss_func)

            # Train
            writer.add_scalar("CIoU/Train", train_output.ciou, global_step=epoch)
            writer.add_scalar("Obj/Train", train_output.obj, global_step=epoch)
            writer.add_scalar("Noobj/Train", train_output.noobj, global_step=epoch)
            writer.add_scalar("Cls/Train", train_output.cls, global_step=epoch)
            writer.add_scalar("Loss/Train", train_output.loss, global_step=epoch)

            # Val
            writer.add_scalar("CIoU/Val", val_output.ciou, global_step=epoch)
            writer.add_scalar("Obj/Val", val_output.obj, global_step=epoch)
            writer.add_scalar("Noobj/Val", val_output.noobj, global_step=epoch)
            writer.add_scalar("Cls/Val", val_output.cls, global_step=epoch)
            writer.add_scalar("Loss/Val", val_output.loss, global_step=epoch)

        torch.save(model.state_dict(), "test-output-60-0.001-0.0001.pth")

if __name__ == "__main__":
    train()