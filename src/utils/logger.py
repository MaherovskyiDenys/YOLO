from torch.utils.tensorboard import SummaryWriter
from src.schema.epoch import EpochSchema

def log_epoch(writer: SummaryWriter, output: EpochSchema, step: int, split: str) -> None:
    """

    :param writer:
    :param output:
    :param step:
    :param split:
    :return:
    """
    writer.add_scalar(f"CIoU/{split}", output.ciou, global_step=step)
    writer.add_scalar(f"Obj/{split}", output.obj, global_step=step)
    writer.add_scalar(f"Noobj/{split}", output.noobj, global_step=step)
    writer.add_scalar(f"Cls/{split}", output.cls, global_step=step)
    writer.add_scalar(f"Loss/{split}", output.loss, global_step=step)

    for idx, (key, value) in enumerate(output.mAP.items()):
        # Removes 'classes' key
        if idx == len(output.mAP.keys()) - 1:
           break

        writer.add_scalar(f"mAP_{split}/{key}", value.item(), global_step=step)

def log_config(writer: SummaryWriter) -> None:
    # Read training.py and write it in tensorboard
    with open("../../configs/training.py", "r+") as file:
        context = file.read()

        writer.add_text("Hyperparameters", f"{context}")