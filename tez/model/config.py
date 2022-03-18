from dataclasses import dataclass
from typing import Optional


@dataclass
class TezConfig:
    experiment_name = "default"
    device: Optional[str] = "cuda"
    learning_rate: Optional[float] = 1e-3
    training_batch_size: Optional[int] = 32
    validation_batch_size: Optional[int] = 32
    epochs: Optional[int] = 20
    gradient_accumulation_steps: Optional[int] = 1
    early_stopping_patience: Optional[int] = 5
    clip_grad_norm: Optional[float] = -1
    fp16: Optional[bool] = False
    num_jobs: Optional[int] = -1
    train_shuffle: Optional[bool] = True
    valid_shuffle: Optional[bool] = True
    train_drop_last: Optional[bool] = False
    valid_drop_last: Optional[bool] = False
    step_scheduler_after: Optional[str] = "epoch"
    step_scheduler_metric: Optional[str] = "current_epoch"
    pin_memory: Optional[bool] = False
    test_batch_size: Optional[int] = 32
    test_drop_last: Optional[bool] = False
    test_shuffle: Optional[bool] = False
