import torch
from dataclasses import dataclass
from src.utils.paths import define_checkpoint_path

@dataclass
class Checkpoint:
    mAP: float = 0.0

    def update(self, model, mAP, run_name):
        if mAP > self.mAP:
            old_mAP = self.mAP
            self.mAP = mAP

            fname = define_checkpoint_path(run_name)
            self._save_checkpoint(model, fname)
            print(f"[Checkpoint Saved] mAP improved from {old_mAP:.4f} -> {self.mAP:.4f} | Filename: {fname}-mAP.pth")

    def _save_checkpoint(self, model, fname):
        torch.save(model.state_dict(), f"{fname}-mAP.pth")
