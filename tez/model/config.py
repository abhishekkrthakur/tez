from dataclasses import dataclass
from typing import Optional


@dataclass
class TezConfig:
    experiment_name = "default"
    device: Optional[str] = "cuda"  # cuda or cpu

    # batch sizes
    training_batch_size: Optional[int] = 32
    validation_batch_size: Optional[int] = 32
    test_batch_size: Optional[int] = 32

    # training parameters
    epochs: Optional[int] = 20
    gradient_accumulation_steps: Optional[int] = 1
    clip_grad_norm: Optional[float] = -1
    num_jobs: Optional[int] = -1
    fp16: Optional[bool] = False

    # data loader parameters
    train_shuffle: Optional[bool] = True
    valid_shuffle: Optional[bool] = True
    train_drop_last: Optional[bool] = False
    valid_drop_last: Optional[bool] = False
    test_drop_last: Optional[bool] = False
    test_shuffle: Optional[bool] = False
    pin_memory: Optional[bool] = True

    # scheduler parameters
    step_scheduler_after: Optional[str] = "epoch"  # "epoch" or "batch"
    step_scheduler_metric: Optional[str] = None

    # TODO: validation parameters
    val_strategy: Optional[str] = "epoch"  # epoch or batch
    val_steps: Optional[int] = 100  # not used if val_strategy is "epoch"
