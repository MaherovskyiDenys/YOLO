from dataclasses import dataclass


@dataclass
class RunningLoss:
    ciou: float = 0.0
    obj: float = 0.0
    noobj: float = 0.0
    cls: float = 0.0
    loss: float = 0.0
    batches: int = 0

    def update(self, loss_output):
        self.ciou += loss_output.ciou.item()
        self.obj += loss_output.obj.item()
        self.noobj += loss_output.noobj.item()
        self.cls += loss_output.cls.item()
        self.loss += loss_output.loss.item()

        self.batches += 1

    def compute(self):
        return {
            "ciou": self.ciou / self.batches,
            "obj": self.obj / self.batches,
            "noobj": self.noobj / self.batches,
            "cls": self.cls / self.batches,
            "loss": self.loss / self.batches,
        }