from torch.utils.tensorboard import SummaryWriter
from src.schema.epoch import EpochSchema
from pathlib import Path
from dataclasses import asdict, dataclass


def log_epoch(writer: SummaryWriter, output_train: EpochSchema, output_val: EpochSchema, step: int) -> None:
    for key in asdict(output_train).keys():
        if key != "mAP":
            writer.add_scalars(
                f"Loss/{key}",
                {
                    "train": getattr(output_train, key),
                    "val": getattr(output_val, key)
                },
                global_step=step
            )

    SKIP_KEYS = {
        "classes",
        "map_per_class",
        "mar_100_per_class",
    }  # some values always produce (-1.)

    for key, train_value in output_train.mAP.items():
        if key in SKIP_KEYS:
            continue

        val_value = output_val.mAP[key]

        writer.add_scalars(
            f"mAP/{key}",
            {
                "train": train_value,
                "val": val_value,
            },
            global_step=step
        )

def log_config(writer: SummaryWriter) -> None:
    # Read training.py and write it in tensorboard
    CONFIG_PATH = (Path(__file__).resolve().parents[2] / "configs" / "training.py")

    context = CONFIG_PATH.read_text("utf-8")

    writer.add_text("Hyperparameters", f"{context}")

