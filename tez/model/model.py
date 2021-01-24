"""
The tez model class
"""

import warnings

import psutil
import torch
import torch.nn as nn
from tez import enums
from tez.callbacks import CallbackRunner
from tez.utils import AverageMeter
from tqdm import tqdm

warnings.filterwarnings("ignore", message=torch.optim.lr_scheduler.SAVE_STATE_WARNING)


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        Instead of inheriting from nn.Module, you import tez and inherit from tez.Model
        """
        super().__init__(*args, **kwargs)
        self.train_loader = None
        self.valid_loader = None
        self.optimizer = None
        self.scheduler = None
        self.step_scheduler_after = None
        self.step_scheduler_metric = None
        self.current_epoch = 0
        self.current_train_step = 0
        self.current_valid_step = 0
        self._model_state = None
        self._train_state = None
        self.device = None
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

    def name_to_metric(self, metric_name):
        if metric_name == "current_epoch":
            return self.current_epoch
        v1 = metric_name.split("_")[0]
        v2 = "_".join(metric_name.split("_")[1:])
        return self.metrics[v1][v2]

    def _init_model(
        self,
        device,
        train_dataset,
        valid_dataset,
        train_sampler,
        valid_sampler,
        train_bs,
        valid_bs,
        n_jobs,
        callbacks,
        fp16,
        train_collate_fn,
        valid_collate_fn,
    ):

        if callbacks is None:
            callbacks = list()

        if n_jobs == -1:
            n_jobs = psutil.cpu_count()

        self.device = device

        if next(self.parameters()).device != self.device:
            self.to(self.device)

        if self.train_loader is None:
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=train_bs,
                num_workers=n_jobs,
                sampler=train_sampler,
                shuffle=True,
                collate_fn=train_collate_fn,
            )
        if self.valid_loader is None:
            if valid_dataset is not None:
                self.valid_loader = torch.utils.data.DataLoader(
                    valid_dataset,
                    batch_size=valid_bs,
                    num_workers=n_jobs,
                    sampler=valid_sampler,
                    shuffle=False,
                    collate_fn=valid_collate_fn,
                )

        if self.optimizer is None:
            self.optimizer = self.fetch_optimizer()

        if self.scheduler is None:
            self.scheduler = self.fetch_scheduler()

        self.fp16 = fp16
        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()

        self._callback_runner = CallbackRunner(callbacks, self)
        self.train_state = enums.TrainingState.TRAIN_START

    def monitor_metrics(self, *args, **kwargs):
        return

    def loss(self, *args, **kwargs):
        return

    def fetch_optimizer(self, *args, **kwargs):
        return

    def fetch_scheduler(self, *args, **kwargs):
        return

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def model_fn(self, data):
        for key, value in data.items():
            data[key] = value.to(self.device)
        if self.fp16:
            with torch.cuda.amp.autocast():
                output, loss, metrics = self(**data)
        else:
            output, loss, metrics = self(**data)
        return output, loss, metrics

    def train_one_step(self, data):
        self.optimizer.zero_grad()
        _, loss, metrics = self.model_fn(data)
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
                    if self.step_scheduler_metric is None:
                        self.scheduler.step()
                    else:
                        step_metric = self.name_to_metric(self.step_scheduler_metric)
                        self.scheduler.step(step_metric)
        return loss, metrics

    def validate_one_step(self, data):
        _, loss, metrics = self.model_fn(data)
        return loss, metrics

    def predict_one_step(self, data):
        output, _, _ = self.model_fn(data)
        return output

    def update_metrics(self, losses, monitor):
        self.metrics[self._model_state.value].update(monitor)
        self.metrics[self._model_state.value]["loss"] = losses.avg

    def train_one_epoch(self, data_loader):
        self.train()
        self.model_state = enums.ModelState.TRAIN
        losses = AverageMeter()
        tk0 = tqdm(data_loader, total=len(data_loader))
        for b_idx, data in enumerate(tk0):
            self.train_state = enums.TrainingState.TRAIN_STEP_START
            loss, metrics = self.train_one_step(data)
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

    def validate_one_epoch(self, data_loader):
        self.eval()
        self.model_state = enums.ModelState.VALID
        losses = AverageMeter()
        tk0 = tqdm(data_loader, total=len(data_loader))
        for b_idx, data in enumerate(tk0):
            self.train_state = enums.TrainingState.VALID_STEP_START
            with torch.no_grad():
                loss, metrics = self.validate_one_step(data)
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

    def process_output(self, output):
        output = output.cpu().detach().numpy()
        return output

    def predict(self, dataset, sampler=None, batch_size=16, n_jobs=1, collate_fn=None):
        if next(self.parameters()).device != self.device:
            self.to(self.device)

        if n_jobs == -1:
            n_jobs = psutil.cpu_count()

        if batch_size == 1:
            n_jobs = 0
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=n_jobs, sampler=sampler, collate_fn=collate_fn, pin_memory=True
        )

        if self.training:
            self.eval()

        final_output = []
        tk0 = tqdm(data_loader, total=len(data_loader))

        for b_idx, data in enumerate(tk0):
            with torch.no_grad():
                out = self.predict_one_step(data)
                out = self.process_output(out)
                yield out

            tk0.set_postfix(stage="test")
        tk0.close()

    def save(self, model_path):
        model_state_dict = self.state_dict()
        if self.optimizer is not None:
            opt_state_dict = self.optimizer.state_dict()
        else:
            opt_state_dict = None
        if self.scheduler is not None:
            sch_state_dict = self.scheduler.state_dict()
        else:
            sch_state_dict = None
        model_dict = {}
        model_dict["state_dict"] = model_state_dict
        model_dict["optimizer"] = opt_state_dict
        model_dict["scheduler"] = sch_state_dict
        model_dict["epoch"] = self.current_epoch
        model_dict["fp16"] = self.fp16
        torch.save(model_dict, model_path)

    def load(self, model_path, device="cuda"):
        self.device = device
        if next(self.parameters()).device != self.device:
            self.to(self.device)
        model_dict = torch.load(model_path)
        self.load_state_dict(model_dict["state_dict"])

    def fit(
        self,
        train_dataset,
        valid_dataset=None,
        train_sampler=None,
        valid_sampler=None,
        device="cuda",
        epochs=10,
        train_bs=16,
        valid_bs=16,
        n_jobs=8,
        callbacks=None,
        fp16=False,
        train_collate_fn=None,
        valid_collate_fn=None,
    ):
        self._init_model(
            device=device,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            train_sampler=train_sampler,
            valid_sampler=valid_sampler,
            train_bs=train_bs,
            valid_bs=valid_bs,
            n_jobs=n_jobs,
            callbacks=callbacks,
            fp16=fp16,
            train_collate_fn=train_collate_fn,
            valid_collate_fn=valid_collate_fn,
        )

        for _ in range(epochs):
            self.train_state = enums.TrainingState.EPOCH_START
            self.train_state = enums.TrainingState.TRAIN_EPOCH_START
            train_loss = self.train_one_epoch(self.train_loader)
            self.train_state = enums.TrainingState.TRAIN_EPOCH_END
            if self.valid_loader:
                self.train_state = enums.TrainingState.VALID_EPOCH_START
                valid_loss = self.validate_one_epoch(self.valid_loader)
                self.train_state = enums.TrainingState.VALID_EPOCH_END
            if self.scheduler:
                if self.step_scheduler_after == "epoch":
                    if self.step_scheduler_metric is None:
                        self.scheduler.step()
                    else:
                        step_metric = self.name_to_metric(self.step_scheduler_metric)
                        self.scheduler.step(step_metric)
            self.train_state = enums.TrainingState.EPOCH_END
            if self._model_state.value == "end":
                break
            self.current_epoch += 1
        self.train_state = enums.TrainingState.TRAIN_END
