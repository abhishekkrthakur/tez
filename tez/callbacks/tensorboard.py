from torch.utils.tensorboard import SummaryWriter

from tez.callbacks import Callback


class TensorBoardLogger(Callback):
    def __init__(self, log_dir=".logs/"):
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=30)

    def on_train_epoch_end(self, model):
        self.writer.add_scalar(
            "train/loss", model.metrics["train"]["loss"], model.current_epoch
        )

    def on_valid_epoch_end(self, model):
        self.writer.add_scalar(
            "valid/loss", model.metrics["valid"]["loss"], model.current_epoch
        )
