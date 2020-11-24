import numpy as np
from tez import enums
from tez.callbacks import Callback

# if import doesn't work:
# pip install sandesh
import sandesh

class SendSandesh(Callback):
    def __init__(self, monitor, patience=5, mode="min", delta=0.001, webhook=None, initial_msg="Hello, this is sandesh! Your model is training...", es_notifications=False):
        self.monitor = monitor
        self.patience = patience

        # sends early stopping notifications if es_notifications = True
        self.es_notifications = es_notifications

        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

        if self.monitor.startswith("train_"):
            self.model_state = "train"
            self.monitor_value = self.monitor[len("train_") :]
        elif self.monitor.startswith("valid_"):
            self.model_state = "valid"
            self.monitor_value = self.monitor[len("valid_") :]
        else:
            raise Exception("monitor must start with train_ or valid_")

        # webhook required to send messages to slack
        self.webhook = webhook

        # welcome message
        try:
            sandesh.send(initial_msg, webhook=self.webhook)
        except:
            raise Exception("please follow the correct syntax: sandesh.send(msg, webhook='XXXX')")

    def on_epoch_end(self, model):
        epoch_score = model.metrics[self.model_state][self.monitor_value]
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score

            # send initial score
            msg = f"Epoch score: {self.best_score}"
            sandesh.send(msg, webhook=self.webhook)
            self.save_checkpoint(epoch_score, model)

        # early stopping notifications
        elif score < self.best_score + self.delta and self.es_notifications == True:
            self.counter += 1
            msg = "EarlyStopping counter: {} out of {}".format(self.counter, self.patience)
            sandesh.send(msg, webhook=self.webhook)
        else:
            self.best_score = score
            msg = f"Epoch score: {self.best_score}"
            sandesh.send(msg, webhook=self.webhook)
            self.save_checkpoint(epoch_score, model)
            self.counter = 0

    # validation score improvement messages
    def save_checkpoint(self, epoch_score, model):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            msg = "Validation score improved ({} --> {}). Saving model!".format(self.val_score, epoch_score)
            sandesh.send(msg, webhook=self.webhook)
        self.val_score = epoch_score
