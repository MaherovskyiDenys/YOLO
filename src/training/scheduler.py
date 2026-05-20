from torch import optim

def build_scheduler(optimizer: optim.Optimizer, step_size=2, gamma=0.1):
    return optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)