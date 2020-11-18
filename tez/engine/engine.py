"""
__author__: Abhishek Thakur
"""

import torch
from tqdm import tqdm

from tez.utils import AverageMeter


class Engine:
    def __init__(self, model, optimizer, device, scheduler=None, accumulation_steps=1):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.accumulation_steps = accumulation_steps

    def train(self, data_loader):
        losses = AverageMeter()
        self.model.train()

        if self.accumulation_steps > 1:
            self.optimizer.zero_grad()

        tk0 = tqdm(data_loader, total=len(data_loader))

        for b_idx, data in enumerate(tk0):
            if self.accumulation_steps == 1 and b_idx == 0:
                self.optimizer.zero_grad()

            for key, value in data.items():
                data[key] = value.to(self.device)
            _, loss = self.model(**data)

            with torch.set_grad_enabled(True):
                loss.backward()
                if (b_idx + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    if b_idx > 0:
                        self.optimizer.zero_grad()

            losses.update(loss.item(), data_loader.batch_size)
            tk0.set_postfix(loss=losses.avg)
        tk0.close()
        return losses.avg

    def evaluate(self, data_loader):
        losses = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            tk0 = tqdm(data_loader, total=len(data_loader))
            for data in tk0:
                for key, value in data.items():
                    data[key] = value.to(self.device)
                _, loss = self.model(**data)
                losses.update(loss.item(), data_loader.batch_size)
                tk0.set_postfix(loss=losses.avg)
            tk0.close()
        return losses.avg

    def predict(self, data_loader):
        self.model.eval()
        final_predictions = []
        with torch.no_grad():
            tk0 = tqdm(data_loader, total=len(data_loader))
            for data in tk0:
                for key, value in data.items():
                    data[key] = value.to(self.device)
                predictions, _ = self.model(**data)
                predictions = predictions.cpu()
                final_predictions.append(predictions)
        return final_predictions
