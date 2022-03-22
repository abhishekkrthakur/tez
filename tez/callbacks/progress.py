from tqdm.auto import tqdm

from tez.callbacks import Callback


class Progress(Callback):
    def __init__(self, num_train_steps, num_valid_steps):
        self.num_train_steps = num_train_steps
        self.num_valid_steps = num_valid_steps
        self._train_tqdm = None

    def on_train_start(self, tez_trainer, **kwargs):
        self._train_tqdm = tqdm(total=self.num_train_steps)

    def format_metrics(self, metrics, stage):
        epoch = metrics["epoch"]
        metrics_str = ", ".join(["{}: {:.4f}".format(k, v) for k, v in metrics.items() if k != "epoch"])
        return f"[{stage} e={epoch}] {metrics_str}"

    def on_train_step_end(self, tez_trainer, **kwargs):
        train_metrics = tez_trainer.metrics["train"]
        # print(train_metrics)
        # self._train_tqdm.set_postfix(epoch=tez_trainer.current_epoch, **train_metrics)
        self._train_tqdm.update(1)

    def on_valid_epoch_end(self, tez_trainer, **kwargs):
        # check if there are any validation metrics
        if len(tez_trainer.metrics["valid"]) == 0:
            return

        train_metrics = tez_trainer.metrics["train"]
        train_metrics["epoch"] = tez_trainer.current_epoch
        # train_metrics["batches"] = tez_trainer.current_train_step
        train_metrics = self.format_metrics(train_metrics, stage="train")
        self._train_tqdm.write(train_metrics)

        valid_metrics = tez_trainer.metrics["valid"]
        valid_metrics["epoch"] = tez_trainer.current_epoch
        # valid_metrics["batches"] = tez_trainer.current_train_step
        valid_metrics = self.format_metrics(valid_metrics, stage="valid")
        self._train_tqdm.write(valid_metrics)

    def on_train_end(self, tez_trainer, **kwargs):
        self._train_tqdm.close()
