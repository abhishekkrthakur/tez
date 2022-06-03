import multiprocessing
import os
import time
import warnings

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from tez import enums
from tez.callbacks import CallbackRunner, Progress
from tez.logger import logger
from tez.utils import AverageMeter

from .config import TezConfig


warnings.filterwarnings("ignore", category=UserWarning)


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
        self.metrics_meter = None
        self.metrics = {}
        self.metrics["train"] = {}
        self.metrics["valid"] = {}
        self.metrics["test"] = {}
        self.num_train_steps = None
        self.num_valid_steps = None

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
                        output_device=self.local_rank,
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

        self.num_train_steps = int(len(self.train_dataset) / self.config.training_batch_size * self.config.epochs)
        if self.valid_dataset:
            self.num_valid_steps = int(len(self.valid_dataset) / self.config.validation_batch_size)
        else:
            self.num_valid_steps = None

        _progress = Progress(num_train_steps=self.num_train_steps, num_valid_steps=self.num_valid_steps)

        if "callbacks" in kwargs:
            self.callbacks = [_progress] + kwargs["callbacks"]
        else:
            self.callbacks = [_progress]

        if self.config.num_jobs == -1:
            self.config.num_jobs = multiprocessing.cpu_count()
            if self.config.num_jobs > 4:
                self.config.num_jobs -= 2

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

        self.optimizer, self.scheduler = self.model.optimizer_scheduler()

        if self.config.fp16:
            self.scaler = torch.cuda.amp.GradScaler()

        self._callback_runner = CallbackRunner(self.callbacks, self)
        self._configure_model()
        self.train_state = enums.TrainingState.TRAIN_START

        if self.optimizer is None:
            raise Exception("No optimizer found")

        if self.local_rank != -1:
            if torch.distributed.get_rank() == 0:
                logger.info(f"\n{self.config}")
                if self.scheduler is None:
                    logger.warning("No scheduler found. Continuing without scheduler")
        else:
            logger.info(f"\n{self.config}")
            if self.scheduler is None:
                logger.warning("No scheduler found. Continuing without scheduler")

    def _init_load_weights(self, config):
        self.config = config
        if self.config.device == "cpu":
            device = torch.device("cpu")
        elif self.config.device == "cuda":
            device = torch.device("cuda:0")
        else:
            raise Exception("Unknown device. Please use 'cpu' or 'cuda'")

        if next(self.model.parameters()).device != device:
            self.model.to(device)

        return device

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
            if self.local_rank != -1:
                if torch.distributed.get_rank() == 0:
                    self._callback_runner(value)
            else:
                self._callback_runner(value)

    def name_to_metric(self, metric_name):
        if metric_name == "current_epoch":
            return self.current_epoch
        v_1 = metric_name.split("_")[0]
        v_2 = "_".join(metric_name.split("_")[1:])
        return self.metrics[v_1][v_2]

    def update_metrics(self, losses, monitor):
        if self._model_state.value == "end":
            return
        self.metrics[self._model_state.value].update(monitor)
        self.metrics[self._model_state.value]["loss"] = losses.avg

    def save(self, model_path, weights_only=False):
        if self.local_rank != -1 or self.num_gpu > 1:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        if weights_only:
            if self.local_rank != -1 and self.num_gpu > 1:
                if torch.distributed.get_rank() == 0:
                    torch.save(model_state_dict, model_path)
            else:
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

        if self.local_rank != -1 and self.num_gpu > 1:
            if torch.distributed.get_rank() == 0:
                torch.save(model_dict, model_path)
        else:
            torch.save(model_dict, model_path)

    def load(self, model_path, weights_only=False, config: TezConfig = None):
        if config is None:
            config = TezConfig()

        device = self._init_load_weights(config)

        model_dict = torch.load(model_path, map_location=device)
        if weights_only:
            self.model.load_state_dict(model_dict)
        else:
            self.model.load_state_dict(model_dict["state_dict"])

    def model_fn(self, data):
        for key, value in data.items():
            data[key] = value.to(self.config.device)
        if self.config.fp16:
            with torch.cuda.amp.autocast():
                output, loss, metrics = self.model(**data)
        else:
            output, loss, metrics = self.model(**data)
        return output, loss, metrics

    def _zero_grad(self):
        if self.config.gradient_accumulation_steps == 1 and self.batch_index == 0:
            self.model.zero_grad()

    def _backward(self, loss, metrics):
        if self.num_gpu > 1:
            loss = loss.mean()
            for metric in metrics:
                metrics[metric] = metrics[metric].mean()
        loss = loss / self.config.gradient_accumulation_steps

        if self.config.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _clip_grad_norm(self):
        if self.config.clip_grad_norm != -1:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)

    def _step(self):
        is_bi_mod_acc_zero = (self.batch_index + 1) % self.config.gradient_accumulation_steps == 0
        is_bi_end = self.batch_index + 1 == self.train_loader
        if is_bi_mod_acc_zero or is_bi_end:
            if self.config.fp16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            if self.scheduler is not None:
                if self.config.step_scheduler_after == "batch":
                    if self.config.step_scheduler_metric is None:
                        self.scheduler.step()
                    else:
                        step_metric = self.name_to_metric(self.config.step_scheduler_metric)
                        self.scheduler.step(step_metric)

            self.model.zero_grad()

    def train_step(self, data):
        self._zero_grad()
        _, loss, metrics = self.model_fn(data)
        self._backward(loss, metrics)
        self._clip_grad_norm()
        self._step()
        return loss, metrics

    def predict_step(self, data):
        _, loss, metrics = self.model_fn(data)
        if self.num_gpu > 1:
            loss = loss.mean()
            for metric in metrics:
                metrics[metric] = metrics[metric].mean()
        return loss, metrics

    def _set_training_epoch_start(self, data_loader):
        if isinstance(data_loader, DataLoader) and isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.set_epoch(self.current_epoch)

        self.model_state = enums.ModelState.TRAIN
        self.train_state = enums.TrainingState.TRAIN_EPOCH_START
        self.model.train()
        if self.config.gradient_accumulation_steps > 1:
            self.optimizer.zero_grad()

    def _update_loss_metrics(self, losses, loss, metrics, data_loader):
        if self.model_state == enums.ModelState.TRAIN:
            losses.update(loss.item() * self.config.gradient_accumulation_steps, data_loader.batch_size)
        else:
            losses.update(loss.item(), data_loader.batch_size)

        if self.batch_index == 0:
            self.metrics_meter = {k: AverageMeter() for k in metrics}

        monitor = {}
        for m_m in self.metrics_meter:
            self.metrics_meter[m_m].update(metrics[m_m].cpu().detach().numpy(), data_loader.batch_size)
            monitor[m_m] = self.metrics_meter[m_m].avg
        if self.model_state == enums.ModelState.TRAIN:
            self.current_train_step += 1
        else:
            self.current_valid_step += 1
        self.update_metrics(losses=losses, monitor=monitor)
        return losses, monitor

    def _set_training_epoch_end(self, losses, monitor):
        self.update_metrics(losses=losses, monitor=monitor)
        self.train_state = enums.TrainingState.TRAIN_EPOCH_END

    def _set_training_state(self):
        self.model_state = enums.ModelState.TRAIN
        self.train_state = enums.TrainingState.TRAIN_EPOCH_START
        self.model.train()

    def train(self, data_loader):
        self._set_training_epoch_start(data_loader)
        losses = AverageMeter()
        for batch_index, data in enumerate(data_loader):
            self.batch_index = batch_index
            self.train_state = enums.TrainingState.TRAIN_STEP_START
            loss, metrics = self.train_step(data)
            losses, monitor = self._update_loss_metrics(losses, loss, metrics, data_loader)
            self.train_state = enums.TrainingState.TRAIN_STEP_END
            if self.valid_loader and self.config.val_strategy == "batch":
                if (
                    self.current_train_step % self.config.val_steps == 0
                    or self.current_train_step == self.num_train_steps
                ):
                    self.validate(self.valid_loader)
            if self._model_state.value == "end":
                break
        self._set_training_epoch_end(losses, monitor)

    def _set_validation_epoch_start(self, data_loader):
        if isinstance(data_loader, DataLoader) and isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.set_epoch(self.current_epoch)

        self.train_state = enums.TrainingState.VALID_EPOCH_START
        self.model_state = enums.ModelState.VALID
        self.model.eval()

    def _set_validation_epoch_end(self, losses, monitor):
        self.update_metrics(losses=losses, monitor=monitor)
        self.train_state = enums.TrainingState.VALID_EPOCH_END

    def validate(self, data_loader):
        self._set_validation_epoch_start(data_loader)
        losses = AverageMeter()

        for batch_index, data in enumerate(data_loader):
            self.batch_index = batch_index
            self.train_state = enums.TrainingState.VALID_STEP_START
            with torch.no_grad():
                loss, metrics = self.predict_step(data)
            losses, monitor = self._update_loss_metrics(losses, loss, metrics, data_loader)
            self.train_state = enums.TrainingState.VALID_STEP_END
        self._set_validation_epoch_end(losses, monitor)
        if self.config.val_strategy == "batch" and self._model_state.value != "end":
            self._set_training_state()

    def _step_scheduler_after_epoch(self):
        if self.scheduler is not None:
            if self.config.step_scheduler_after == "epoch":
                if self.config.step_scheduler_metric is None:
                    self.scheduler.step()
                else:
                    step_metric = self.name_to_metric(self.config.step_scheduler_metric)
                    self.scheduler.step(step_metric)

    def fit(self, train_dataset, valid_dataset=None, config: TezConfig = None, **kwargs):
        if config is None:
            config = TezConfig()
        self._init_trainer(train_dataset, valid_dataset, config, **kwargs)

        for _ in range(self.config.epochs):
            self.train_state = enums.TrainingState.EPOCH_START
            self.train(self.train_loader)
            if self.valid_loader and self.config.val_strategy == "epoch":
                self.validate(self.valid_loader)
            self._step_scheduler_after_epoch()
            self.train_state = enums.TrainingState.EPOCH_END
            if self._model_state.value == "end":
                time.sleep(2)
                break
            self.current_epoch += 1
        self.train_state = enums.TrainingState.TRAIN_END
        # TODO: do we need this?
        # if self.local_rank != -1:
        #    torch.distributed.barrier()

    def process_output(self, output):
        output = output.cpu().detach().numpy()
        return output

    def predict(self, dataset, **kwargs):

        if "sampler" in kwargs:
            sampler = kwargs["sampler"]
        else:
            sampler = None

        if "collate_fn" in kwargs:
            collate_fn = kwargs["collate_fn"]
        else:
            collate_fn = None

        if "batch_size" in kwargs:
            batch_size = kwargs["batch_size"]
        else:
            batch_size = self.config.test_batch_size

        if "num_jobs" in kwargs:
            num_jobs = kwargs["num_jobs"]
        else:
            num_jobs = self.config.num_jobs

        if "pin_memory" in kwargs:
            pin_memory = kwargs["pin_memory"]
        else:
            pin_memory = self.config.pin_memory

        if num_jobs == -1:
            num_jobs = multiprocessing.cpu_count()
            if num_jobs > 4:
                num_jobs -= 2

        if batch_size == 1:
            num_jobs = 0

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_jobs,
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )

        if self.model.training:
            self.model.eval()

        for data in data_loader:
            with torch.no_grad():
                out, _, _ = self.model_fn(data)
                out = self.process_output(out)
                yield out
