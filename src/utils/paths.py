import socket
from datetime import datetime
import os
from pathlib import Path
from typing import Optional

def define_run_name(name: Optional[str] = None) -> str:
    base = Path(__file__).resolve().parents[2]

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(current_time + "_" + socket.gethostname())

    return (base / "runs" / log_dir).as_posix() if not name else (base / "runs" / name).as_posix()


def define_checkpoint_path(run_name: str) -> str:
    return (Path(__file__).resolve().parents[2] / "models" / Path(run_name).name).as_posix()