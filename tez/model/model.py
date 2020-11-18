from abc import ABC, abstractmethod

import psutil
import torch
import torch.nn as nn

from tez.utils import AverageMeter
from tqdm import tqdm


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.valid_loader = None
        self.step_scheduler_after = None
        self.metrics = None

    def monitor_metrics(self, outputs, targets, *args, **kwargs):
        if outputs is None:
            return None
        if targets is None:
            return None
        return None

    def create_scheduler(self, step_after, *args, **kwargs):
        if step_after not in ("batch", "epoch"):
            raise Exception("step parameter should be either batch or epoch")
        self.step_scheduler_after = step_after

    def create_optimizer(self, *args, **kwargs):
        return

    def loss(self, *args, **kwargs):
        return

    def forward(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    def train_one_step(self, data, device):
        for key, value in data.items():
            data[key] = value.to(device)
        _, loss = self(**data)
        return loss

    def validate_one_step(self, data, device):
        for key, value in data.items():
            data[key] = value.to(device)
        _, loss = self(**data)
        return loss

    def train_one_epoch(self, data_loader, device):
        self.train()
        losses = AverageMeter()
        tk0 = tqdm(data_loader, total=len(data_loader))
        for data in tk0:
            self.optimizer.zero_grad()
            loss = self.train_one_step(data, device)
            with torch.set_grad_enabled(True):
                loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    if self.step_scheduler_after == "batch":
                        self.scheduler.step()
            losses.update(loss.item(), data_loader.batch_size)
            tk0.set_postfix(loss=losses.avg, stage="train")
        tk0.close()
        return losses.avg

    def validate_one_epoch(self, data_loader, device):
        self.eval()
        losses = AverageMeter()
        tk0 = tqdm(data_loader, total=len(data_loader))
        for data in tk0:
            with torch.no_grad():
                loss = self.train_one_step(data, device)
            losses.update(loss.item(), data_loader.batch_size)
            tk0.set_postfix(loss=losses.avg, stage="valid")
        tk0.close()
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
    ):

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

        for _ in range(epochs):
            train_loss = self.train_one_epoch(self.train_loader, device)
            if self.valid_loader:
                valid_loss = self.validate_one_epoch(self.valid_loader, device)

            if self.scheduler:
                if self.step_scheduler_after == "epoch":
                    self.scheduler.step()
