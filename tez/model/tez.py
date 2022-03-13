import multiprocessing
import os
import warnings

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from tez import enums
from tez.callbacks import CallbackRunner
from tez.logger import logger
from tez.utils import AverageMeter

from .config import TezConfig


warnings.filterwarnings("ignore", category=UserWarning)
g

class Tez:
    def __init__(self, model):
        self.model = model
        self.config = None
        self.train_dataset = None
        self.valid_dataset = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.current_epoch = 0
        self.batch_index = 0
        self.current_train_step = 0
        self.current_valid_step = 0
        self.num_gpu = 0
        self.local_rank = -1
        self.world_size = 1
        self._model_state = None
        self._train_state = None
        self.metrics = {}
        self.metrics["train"] = {}
        self.metrics["valid"] = {}
        self.metrics["test"] = {}

    def _configure_model(self):
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank != -1 and local_rank != self.local_rank:
            self.local_rank = local_rank

        if self.config.device == "cpu":
            device = torch.device("cpu")
            self.num_gpu = 0
        elif self.config.device == "cuda":
            if torch.cuda.device_count() > 1:
                if self.local_rank == -1:
                    device = torch.device("cuda:0")
                    self.num_gpu = torch.cuda.device_count()
                    self.model = torch.nn.DataParallel(self.model, device_ids=list(range(self.num_gpu)))
                else:
                    torch.distributed.init_process_group(backend="nccl")
                    self.world_size = torch.distributed.get_world_size()
                    device = torch.device("cuda", self.local_rank)
                    self.model.to(device)
                    self.model = torch.nn.parallel.DistributedDataParallel(
                        self.model,
                        device_ids=[self.local_rank],
                    )
                    self.num_gpu = 1
            else:
                logger.info("Using single GPU")
                device = torch.device("cuda:0")
                self.num_gpu = 1

        if self.local_rank == -1:
            self.model.to(device)

    def _init_trainer(self, train_dataset, valid_dataset, config, **kwargs):
        self.config = config
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        if "train_loader" in kwargs:
            self.train_loader = kwargs["train_loader"]
        else:
            self.train_loader = None
        if "valid_loader" in kwargs:
            self.valid_loader = kwargs["valid_loader"]
        else:
            self.valid_loader = None

        if "train_sampler" in kwargs:
            self.train_sampler = kwargs["train_sampler"]
        else:
            self.train_sampler = None

        if "valid_sampler" in kwargs:
            self.valid_sampler = kwargs["valid_sampler"]
        else:
            self.valid_sampler = None

        if "train_collate_fn" in kwargs:
            self.train_collate_fn = kwargs["train_collate_fn"]
        else:
            self.train_collate_fn = None

        if "valid_collate_fn" in kwargs:
            self.valid_collate_fn = kwargs["valid_collate_fn"]
        else:
            self.valid_collate_fn = None

        if "callbacks" in kwargs:
            self.callbacks = kwargs["callbacks"]
        else:
            self.callbacks = []

        if self.config.num_jobs == -1:
            self.config.num_jobs = multiprocessing.cpu_count()

        if self.world_size > 1 and self.train_sampler is None:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
            )

        if self.world_size > 1 and self.valid_sampler is None:
            self.valid_sampler = DistributedSampler(
                self.valid_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
            )

        if self.train_loader is None:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.training_batch_size,
                num_workers=self.config.num_jobs,
                sampler=self.train_sampler,
                shuffle=self.config.train_shuffle,
                collate_fn=self.train_collate_fn,
                drop_last=self.config.train_drop_last,
                pin_memory=self.config.pin_memory,
            )

        if self.valid_loader is None:
            if self.valid_dataset is not None:
                self.valid_loader = DataLoader(
                    self.valid_dataset,
                    batch_size=self.config.validation_batch_size,
                    num_workers=self.config.num_jobs,
                    sampler=self.valid_sampler,
                    shuffle=self.config.valid_shuffle,
                    collate_fn=self.valid_collate_fn,
                    drop_last=self.config.valid_drop_last,
                    pin_memory=self.config.pin_memory,
                )

        self.optimizer = self.model.fetch_optimizer()
        try:
            self.scheduler = self.model.fetch_scheduler()
        except AttributeError:
            logger.warning("No scheduler found")

        if self.config.fp16:
            self.scaler = torch.cuda.amp.GradScaler()

        self._callback_runner = CallbackRunner(self.callbacks, self)
        self._configure_model()
        self.train_state = enums.TrainingState.TRAIN_START

    @property
    def model_state(self):
        return self._model_state

    @model_state.setter
    def model_state(self, value):
        self._model_state = value

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
        v_1 = metric_name.split("_")[0]
        v_2 = "_".join(metric_name.split("_")[1:])
        return self.metrics[v_1][v_2]

    def update_metrics(self, losses, monitor):
        self.metrics[self._model_state.value].update(monitor)
        self.metrics[self._model_state.value]["loss"] = losses.avg

    def save(self, model_path, weights_only=False):
        if self.local_rank != -1 or self.num_gpu > 1:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        if weights_only:
            torch.save(model_state_dict, model_path)
            return

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
        model_dict["config"] = self.config
        torch.save(model_dict, model_path)

    def load(self, model_path, weights_only=False):
        # if next(self.model.parameters()).device != self.device:
        #    self.to(self.device)
        # model_dict = torch.load(model_path, map_location=torch.device(device))
        model_dict = torch.load(model_path)
        if weights_only:
            self.load_state_dict(model_dict)
        else:
            self.load_state_dict(model_dict["state_dict"])

    def model_fn(self, data):
        for key, value in data.items():
            data[key] = value.to(self.config.device)
        if self.config.fp16:
            with torch.cuda.amp.autocast():
                output, loss, metrics = self.model(**data)
        else:
            output, loss, metrics = self.model(**data)
        return output, loss, metrics

    def train_step(self, data):
        if self.config.gradient_accumulation_steps == 1 and self.batch_index == 0:
            self.model.zero_grad()
        _, loss, metrics = self.model_fn(data)
        if self.num_gpu > 1:
            loss = loss.mean()
            for metric in metrics:
                metrics[metric] = metrics[metric].mean()
        loss = loss / self.config.gradient_accumulation_steps

        if self.config.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if self.config.clip_grad_norm != -1:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)

        if (self.batch_index + 1) % self.config.gradient_accumulation_steps == 0:
            if self.config.fp16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            if self.scheduler:
                if self.config.step_scheduler_after == "batch":
                    if self.config.step_scheduler_metric is None:
                        self.scheduler.step()
                    else:
                        step_metric = self.name_to_metric(self.config.step_scheduler_metric)
                        self.scheduler.step(step_metric)

            if self.batch_index > 0:
                self.model.zero_grad()

        return loss, metrics

    def predict_step(self, data):
        _, loss, metrics = self.model_fn(data)
        if self.num_gpu > 1:
            loss = loss.mean()
            for metric in metrics:
                metrics[metric] = metrics[metric].mean()
        return loss, metrics

    def train(self, data_loader, _tqdm=None):
        if isinstance(data_loader, DataLoader) and isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.set_epoch(self.current_epoch)

        self.train_state = enums.TrainingState.EPOCH_START
        self.train_state = enums.TrainingState.TRAIN_EPOCH_START
        self.model.train()
        self.model_state = enums.ModelState.TRAIN
        losses = AverageMeter()
        if self.config.gradient_accumulation_steps > 1:
            self.optimizer.zero_grad()

        for batch_index, data in enumerate(data_loader):
            self.batch_index = batch_index
            self.train_state = enums.TrainingState.TRAIN_STEP_START
            loss, metrics = self.train_step(data)
            self.train_state = enums.TrainingState.TRAIN_STEP_END
            losses.update(loss.item() * self.config.gradient_accumulation_steps, data_loader.batch_size)
            if batch_index == 0:
                metrics_meter = {k: AverageMeter() for k in metrics}
            monitor = {}
            for m_m in metrics_meter:
                metrics_meter[m_m].update(metrics[m_m].cpu().detach().numpy(), data_loader.batch_size)
                monitor[m_m] = metrics_meter[m_m].avg
            self.current_train_step += 1
            if _tqdm is not None:
                _tqdm.set_postfix(loss=losses.avg, stage="train", epoch=self.current_epoch, **monitor)
                _tqdm.update(1)
        self.update_metrics(losses=losses, monitor=monitor)
        self.train_state = enums.TrainingState.TRAIN_EPOCH_END
        return losses.avg

    def validate(self, data_loader, _tqdm=None):
        if isinstance(data_loader, DataLoader) and isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.set_epoch(self.current_epoch)
        self.train_state = enums.TrainingState.VALID_EPOCH_START
        self.model.eval()
        self.model_state = enums.ModelState.VALID
        losses = AverageMeter()

        for batch_index, data in enumerate(data_loader):
            self.train_state = enums.TrainingState.VALID_STEP_START
            with torch.no_grad():
                loss, metrics = self.predict_step(data)
            self.train_state = enums.TrainingState.VALID_STEP_END
            losses.update(loss.item(), data_loader.batch_size)
            if batch_index == 0:
                metrics_meter = {k: AverageMeter() for k in metrics}
            monitor = {}
            for m_m in metrics_meter:
                metrics_meter[m_m].update(metrics[m_m].cpu().detach().numpy(), data_loader.batch_size)
                monitor[m_m] = metrics_meter[m_m].avg
            if _tqdm is not None:
                _tqdm.set_postfix(loss=losses.avg, stage="valid", epoch=self.current_epoch, **monitor)
            self.current_valid_step += 1
        self.update_metrics(losses=losses, monitor=monitor)
        self.train_state = enums.TrainingState.VALID_EPOCH_END
        return losses.avg

    def fit(self, train_dataset, valid_dataset=None, config: TezConfig = None, **kwargs):
        if config is None:
            config = TezConfig()
        logger.info(f"\n{config}")
        self._init_trainer(train_dataset, valid_dataset, config, **kwargs)
        num_train_steps = int(
            len(self.train_dataset)
            / self.config.training_batch_size
            / self.config.gradient_accumulation_steps
            * self.config.epochs
        )
        _tqdm = tqdm(total=num_train_steps)
        for _ in range(self.config.epochs):
            _ = self.train(self.train_loader, _tqdm)
            if self.valid_loader:
                _ = self.validate(self.valid_loader)

            if self.scheduler:
                if self.config.step_scheduler_after == "epoch":
                    if self.config.step_scheduler_metric is None:
                        self.scheduler.step()
                    else:
                        step_metric = self.name_to_metric(self.config.step_scheduler_metric)
                        self.scheduler.step(step_metric)
            self.train_state = enums.TrainingState.EPOCH_END
            if self._model_state.value == "end":
                break
            self.current_epoch += 1
        if self.local_rank != -1:
            torch.distributed.barrier()
        self.train_state = enums.TrainingState.TRAIN_END

    def process_output(self, output):
        output = output.cpu().detach().numpy()
        return output

    def predict(self, dataset, **kwargs):
        if "sampler" in kwargs:
            sampler = kwargs["sampler"]
        else:
            sampler = 16

        if "collate_fn" in kwargs:
            collate_fn = kwargs["collate_fn"]
        else:
            collate_fn = None

        if "batch_size" in kwargs:
            batch_size = kwargs["batch_size"]
        else:
            batch_size = 16

        if "n_jobs" in kwargs:
            n_jobs = kwargs["n_jobs"]
        else:
            n_jobs = -1

        if "pin_memory" in kwargs:
            pin_memory = kwargs["pin_memory"]
        else:
            pin_memory = True
        # if next(self.model.parameters()).device != self.device:
        #    self.model.to(self.device)

        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        if batch_size == 1:
            n_jobs = 0
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=n_jobs,
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )

        if self.model.training:
            self.model.eval()

        tk0 = tqdm(data_loader, total=len(data_loader))

        for _, data in enumerate(tk0):
            with torch.no_grad():
                out = self.predict_step(data)
                out = self.process_output(out)
                yield out

            tk0.set_postfix(stage="test")

        tk0.close()
