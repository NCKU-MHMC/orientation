from typing import List, Optional, Union
from pathlib import Path

from dataclasses import dataclass


@dataclass
class DataConfig:
    train_paths: List[Union[str, Path]]
    valid_paths: List[Union[str, Path]]
    test_paths: List[Union[str, Path]]
    train_batch_size: int
    valid_batch_size: int
    test_batch_size: int

    num_workers: int

    augment: bool


@dataclass
class ModelConfig:
    nblock: int

@dataclass
class OptimArgs:
    lr: float

@dataclass
class OptimConfig:
    optim_type: str
    optim_args: OptimArgs
    regul_lambda: float
    regul_ord: Optional[int]


@dataclass
class HW1Config:
    mode: str
    total_steps: int
    resume_ckpt: Optional[Union[int, str]]
    print_every: int
    save_every: int
    valid_every: int
    ckpt_dir: Union[Path, str]
    test_ckpt: Union[str, int]
    device_id: int

    data: DataConfig
    # model: ModelConfig
    optim: OptimConfig