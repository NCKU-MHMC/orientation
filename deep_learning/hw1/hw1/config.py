from typing import List
from pathlib import Path

from dataclasses import dataclass


@dataclass
class DataConfig:
    train_paths: List[str|Path]
    valid_paths: List[str|Path]
    test_paths: List[str|Path]
    train_batch_size: int
    valid_batch_size: int
    test_batch_size: int


@dataclass
class ModelConfig:
    nblock: int

@dataclass
class OptimConfig:
    optim_type: str

@dataclass
class HW1Config:
    mode: str
    total_iters: int
    resume_iter: int | None
    print_every: int
    save_every: int
    valid_every: int
    ckpt_dir: Path | str
    test_ckpt: str | int

    data: DataConfig
    # model: ModelConfig
    # optim: OptimConfig