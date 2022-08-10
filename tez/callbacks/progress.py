from tqdm.auto import tqdm

from tez.callbacks import Callback


class Progress(Callback):
    def __init__(self, num_train_steps, num_valid_steps):
        self.num_train_steps = num_train_steps
        self.num_valid_steps = num_valid_steps
        self._train_tqdm = None
        self._valid_tqdm = None
        self.history = []

    def on_train_start(self, tez_trainer, **kwargs):
        self._train_tqdm = tqdm(total=self.num_train_steps, disable=not tez_trainer._driver.is_main_process)
        if self.num_valid_steps:
            self._valid_tqdm = tqdm(total=self.num_valid_steps, disable=not tez_trainer._driver.is_main_process)

    def on_valid_epoch_start(self, tez_trainer, **kwargs):
        if self.num_valid_steps:
            self._valid_tqdm = tqdm(total=self.num_valid_steps, leave=False)

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

    def on_valid_step_end(self, tez_trainer, **kwargs):
        if self.num_valid_steps is None:
            return
        self._valid_tqdm.update(1)

    def on_valid_epoch_end(self, tez_trainer, **kwargs):
        if self._valid_tqdm is not None:
            self._valid_tqdm.close()

        if len(tez_trainer.metrics["valid"]) == 0:
            return

        train_metrics = tez_trainer.metrics["train"]
        epoch = tez_trainer.current_epoch
        steps = tez_trainer._train_step
        train_metrics = self.format_metrics(train_metrics, stage="train")

        valid_metrics = tez_trainer.metrics["valid"]
        valid_metrics["epoch"] = tez_trainer.current_epoch
        valid_metrics = self.format_metrics(valid_metrics, stage="valid")

        metrics_string = f"{train_metrics} {valid_metrics} [e={epoch} steps={steps}]"
        if tez_trainer._driver.is_local_main_process:
            self._train_tqdm.write(metrics_string)

        metrics = {}
        metrics["epoch"] = epoch
        metrics["steps"] = steps
        metrics["train"] = {k: v for k, v in tez_trainer.metrics["train"].items() if k not in ("epoch", "steps")}
        metrics["valid"] = {k: v for k, v in tez_trainer.metrics["valid"].items() if k not in ("epoch", "steps")}
        self.history.append(metrics)

    def on_train_epoch_end(self, tez_trainer, **kwargs):
        if self._valid_tqdm is not None:
            return

        train_metrics = tez_trainer.metrics["train"]
        epoch = tez_trainer.current_epoch
        steps = tez_trainer._train_step
        train_metrics = self.format_metrics(train_metrics, stage="train")

        metrics_string = f"{train_metrics} [e={epoch} steps={steps}]"
        if tez_trainer._driver.is_local_main_process:
            self._train_tqdm.write(metrics_string)

        metrics = {}
        metrics["epoch"] = epoch
        metrics["steps"] = steps
        metrics["train"] = {k: v for k, v in tez_trainer.metrics["train"].items() if k not in ("epoch", "steps")}
        self.history.append(metrics)

    def on_train_end(self, tez_trainer, **kwargs):
        self._train_tqdm.close()
