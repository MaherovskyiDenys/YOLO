import torch
from dataclasses import dataclass
from src.utils.paths import define_checkpoint_path

@dataclass
class Checkpoint:
    mAP: float = 0.0

    def update(self, model, optimizer, mAP, run_name):
        if mAP > self.mAP:
            old_mAP = self.mAP
            self.mAP = mAP

            fname = define_checkpoint_path(run_name)
            self._save_checkpoint(model, optimizer, fname)
            print(f"[Checkpoint Saved] mAP improved from {old_mAP:.4f} -> {self.mAP:.4f} | Filename: {fname}-mAP.pth")

    @staticmethod
    def _save_checkpoint(model, optimizer, fname):
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, f"{fname}-mAP.pth")
