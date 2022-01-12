from dataclasses import dataclass
from typing import Optional


@dataclass
class Args:
    train_batch_size: int
    epochs: int
    device: Optional[str] = "cuda"
    fp16: Optional[bool] = False
    valid_batch_size: Optional[int] = None
    n_jobs: Optional[int] = -1
    train_shuffle: Optional[bool] = True
    valid_shuffle: Optional[bool] = False
    accumulation_steps: Optional[int] = 1
    clip_grad_norm: Optional[float] = None
