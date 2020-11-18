import torch
from tez.engine import Engine


class Tez:
    def __init__(self, dataset, model, batch_size, **kwargs):
        self.model = model
        self.dataset = dataset
        self.device = "gpu"

        self.optimizer = kwargs.get("optimizer", None)
        self.num_workers = kwargs.get("num_workers", None)
        self.scheduler = kwargs.get("scheduler", None)
        self.accumulation_steps = kwargs.get("accumulation_steps", 1)

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
        )

    def run(self, train=True):
        if train is True:
            if self.optimizer is None:
                raise Exception("optimizer must be defined in training mode")
            engine = Engine(
                model=self.model,
                optimizer=self.optimizer,
                device=self.device,
                scheduler=self.scheduler,
                accumulation_steps=self.accumulation_steps,
            )
            engine.train(self.data_loader)
        else:
            engine = Engine(
                model=self.model,
                optimizer=self.optimizer,
                device=self.device,
                scheduler=self.scheduler,
                accumulation_steps=self.accumulation_steps,
            )
            engine.train(self.data_loader)
