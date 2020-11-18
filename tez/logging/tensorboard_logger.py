from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, loc=".logs"):
        self.writer = SummaryWriter(loc, flush_secs=30)

    def log(self, name, value, itr):
        self.writer.add_scalar(name, value, itr)

    def log_image(self, name, img):
        self.writer.add_image(name, img)
