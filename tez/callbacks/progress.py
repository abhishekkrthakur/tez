from tqdm.auto import tqdm

from tez.callbacks import Callback


class Progress(Callback):
    def __init__(self, num_train_steps, num_valid_steps):
        self.num_train_steps = num_train_steps
        self.num_valid_steps = num_valid_steps
        self._train_tqdm = None
        self._valid_tqdm = None

    def on_train_start(self, tez_trainer, **kwargs):
        self._train_tqdm = tqdm(total=self.num_train_steps)
        # if self.num_valid_steps:
        #    self._valid_tqdm = tqdm(total=self.num_valid_steps, leave=False)

    def format_metrics(self, metrics, stage):
        metrics_str = ", ".join(["{}={:.4f}".format(k, v) for k, v in metrics.items() if k not in ("epoch", "steps")])
        if stage == "train":
            return f"[{stage}] {metrics_str}"
        elif stage == "valid":
            return f"[{stage}] {metrics_str}"
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def on_train_step_end(self, tez_trainer, **kwargs):
        train_metrics = tez_trainer.metrics["train"]
        if "epoch" in train_metrics:
            del train_metrics["epoch"]
        self._train_tqdm.set_postfix(epoch=tez_trainer.current_epoch, **train_metrics)
        self._train_tqdm.update(1)

    # def on_valid_step_end(self, tez_trainer, **kwargs):
    #     if self.num_valid_steps is None:
    #         return
    #     self._valid_tqdm.update(1)

    def on_valid_epoch_end(self, tez_trainer, **kwargs):
        # check if there are any validation metrics
        if len(tez_trainer.metrics["valid"]) == 0:
            return

        train_metrics = tez_trainer.metrics["train"]
        epoch = tez_trainer.current_epoch
        steps = tez_trainer.current_train_step
        # train_metrics["batches"] = tez_trainer.current_train_step
        train_metrics = self.format_metrics(train_metrics, stage="train")
        # self._train_tqdm.write(train_metrics)

        valid_metrics = tez_trainer.metrics["valid"]
        valid_metrics["epoch"] = tez_trainer.current_epoch
        # valid_metrics["batches"] = tez_trainer.current_train_step
        valid_metrics = self.format_metrics(valid_metrics, stage="valid")

        metrics_string = f"{train_metrics} {valid_metrics} [e={epoch} steps={steps}]"
        self._train_tqdm.write(metrics_string)

    def on_train_end(self, tez_trainer, **kwargs):
        self._train_tqdm.close()
