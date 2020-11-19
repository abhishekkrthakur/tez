import psutil
import torch
import torch.nn as nn
from tez.utils import AverageMeter
from tqdm.auto import tqdm
from tez import enums
from tez.callbacks import CallbackRunner


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.valid_loader = None
        self.step_scheduler_after = None
        self.current_epoch = 0
        self.current_train_step = 0
        self.current_valid_step = 0
        self._model_state = None
        self._train_state = None
        self._callback_runner = None
        self.fp16 = False
        self.scaler = None
        self.metrics = {}
        self.metrics["train"] = {}
        self.metrics["valid"] = {}
        self.metrics["test"] = {}

    @property
    def model_state(self):
        return self._model_state

    @model_state.setter
    def model_state(self, value):
        self._model_state = value
        # run something here in future if needed

    @property
    def train_state(self):
        return self._train_state

    @train_state.setter
    def train_state(self, value):
        self._train_state = value
        if self._callback_runner is not None:
            self._callback_runner(value)

    def _init_model(
        self,
        device,
        train_dataset,
        valid_dataset,
        train_bs,
        valid_bs,
        n_jobs,
        callbacks,
        fp16,
    ):

        if callbacks is None:
            callbacks = list()

        if n_jobs == -1:
            n_jobs = psutil.cpu_count()

        if next(self.parameters()).device != device:
            self.to(device)

        if self.optimizer is None:
            self.optimizer = self.create_optimizer()

        if self.scheduler is None:
            self.scheduler = self.create_scheduler()

        if self.train_loader is None:
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_bs, num_workers=n_jobs
            )
        if self.valid_loader is None:
            if valid_dataset is not None:
                self.valid_loader = torch.utils.data.DataLoader(
                    valid_dataset, batch_size=valid_bs, num_workers=n_jobs
                )

        self.fp16 = fp16
        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()

        self._callback_runner = CallbackRunner(callbacks, self)
        self.train_state = enums.TrainingState.TRAIN_START

    def create_scheduler(self, step_after, *args, **kwargs):
        if step_after not in ("batch", "epoch"):
            raise Exception("step parameter should be either batch or epoch")
        self.step_scheduler_after = step_after

    def monitor_metrics(self, *args, **kwargs):
        return

    def create_optimizer(self, *args, **kwargs):
        return

    def loss(self, *args, **kwargs):
        return

    def forward(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    def train_one_step(self, data, device):
        self.optimizer.zero_grad()
        for key, value in data.items():
            data[key] = value.to(device)
        if self.fp16:
            with torch.cuda.amp.autocast():
                _, loss, metrics = self(**data)
        else:
            _, loss, metrics = self(**data)
        with torch.set_grad_enabled(True):
            if self.fp16:
                with torch.cuda.amp.autocast():
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            if self.scheduler:
                if self.step_scheduler_after == "batch":
                    self.scheduler.step()
        return loss, metrics

    def validate_one_step(self, data, device):
        for key, value in data.items():
            data[key] = value.to(device)
        _, loss, metrics = self(**data)
        return loss, metrics

    def update_metrics(self, losses, monitor):
        self.metrics[self._model_state.value].update(monitor)
        self.metrics[self._model_state.value]["loss"] = losses.avg

    def train_one_epoch(self, data_loader, device):
        self.train()
        self.model_state = enums.ModelState.TRAIN
        losses = AverageMeter()
        tk0 = tqdm(data_loader, total=len(data_loader))
        for b_idx, data in enumerate(tk0):
            self.train_state = enums.TrainingState.TRAIN_STEP_START
            loss, metrics = self.train_one_step(data, device)
            self.train_state = enums.TrainingState.TRAIN_STEP_END
            losses.update(loss.item(), data_loader.batch_size)
            if b_idx == 0:
                metrics_meter = {k: AverageMeter() for k in metrics}
            monitor = {}
            for m_m in metrics_meter:
                metrics_meter[m_m].update(metrics[m_m], data_loader.batch_size)
                monitor[m_m] = metrics_meter[m_m].avg
            self.current_train_step += 1
            tk0.set_postfix(loss=losses.avg, stage="train", **monitor)
        tk0.close()
        self.update_metrics(losses=losses, monitor=monitor)
        return losses.avg

    def validate_one_epoch(self, data_loader, device):
        self.eval()
        self.model_state = enums.ModelState.VALID
        losses = AverageMeter()
        tk0 = tqdm(data_loader, total=len(data_loader))
        for b_idx, data in enumerate(tk0):
            self.train_state = enums.TrainingState.VALID_STEP_START
            with torch.no_grad():
                loss, metrics = self.validate_one_step(data, device)
            self.train_state = enums.TrainingState.VALID_STEP_END
            losses.update(loss.item(), data_loader.batch_size)
            if b_idx == 0:
                metrics_meter = {k: AverageMeter() for k in metrics}
            monitor = {}
            for m_m in metrics_meter:
                metrics_meter[m_m].update(metrics[m_m], data_loader.batch_size)
                monitor[m_m] = metrics_meter[m_m].avg
            tk0.set_postfix(loss=losses.avg, stage="valid", **monitor)
            self.current_valid_step += 1
        tk0.close()
        self.update_metrics(losses=losses, monitor=monitor)
        return losses.avg

    def fit(
        self,
        train_dataset,
        valid_dataset=None,
        device="cuda",
        epochs=10,
        train_bs=16,
        valid_bs=16,
        n_jobs=8,
        callbacks=None,
        fp16=False,
    ):
        """
        When using fp16, you must wrap loss calculation with the following

        with torch.cuda.amp.autocast():
            loss = model(data)
        """
        self._init_model(
            device=device,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            train_bs=train_bs,
            valid_bs=valid_bs,
            n_jobs=n_jobs,
            callbacks=callbacks,
            fp16=fp16,
        )

        for _ in range(epochs):
            self.train_state = enums.TrainingState.TRAIN_EPOCH_START
            train_loss = self.train_one_epoch(self.train_loader, device)
            self.train_state = enums.TrainingState.TRAIN_EPOCH_END
            if self.valid_loader:
                self.train_state = enums.TrainingState.VALID_EPOCH_START
                valid_loss = self.validate_one_epoch(self.valid_loader, device)
                self.train_state = enums.TrainingState.VALID_EPOCH_END

            if self.scheduler:
                if self.step_scheduler_after == "epoch":
                    self.scheduler.step()
            self.current_epoch += 1
        self.train_state = enums.TrainingState.TRAIN_END
