from dataclasses import asdict
from pathlib import Path

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from configs.training import TEST_BATCH_SIZE
from src.schema.epoch import EpochSchema


def log_epoch(
        writer: SummaryWriter,
        output_train: EpochSchema,
        output_test: EpochSchema,
        step: int
) -> None:
    for key in asdict(output_train).keys():
        if key != "mAP":
            writer.add_scalars(
                f"Loss/{key}",
                {
                    "train": getattr(output_train, key),
                    "test": getattr(output_test, key)
                },
                global_step=step
            )

    # Skipping the per-class arrays, some values always produce (-1.)
    SKIP_KEYS = {
        "classes",
        "map_per_class",
        "mar_100_per_class",
    }

    for key, train_value in output_train.mAP.items():
        if key in SKIP_KEYS:
            continue

        test_value = output_test.mAP[key]

        writer.add_scalars(
            f"mAP/{key}",
            {
                "train": train_value.item(),
                "test": test_value.item(),
            },
            global_step=step
        )

def log_config(writer: SummaryWriter) -> None:
    # Read training.py and write it in tensorboard
    CONFIG_PATH = (Path(__file__).resolve().parents[2] / "configs" / "training.py")

    context = CONFIG_PATH.read_text("utf-8")

    writer.add_text("Hyperparameters", f"{context}")

def log_images(writer: SummaryWriter, images: Tensor, step):
    """Images already with bounding boxes labels and scores"""
    writer.add_images(f'Validation: {TEST_BATCH_SIZE} imgs', images, step)
