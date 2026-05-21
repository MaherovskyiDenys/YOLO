from pathlib import Path

def define_run_name(name: str) -> str:
    return (Path(__file__).resolve().parents[2] / "runs" / name).as_posix()
