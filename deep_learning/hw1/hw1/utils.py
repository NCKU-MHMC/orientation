import torch
from torch import optim
from torch.nn import Parameter

from hw1.config import OptimConfig

from typing import Iterator, Optional, TypeVar, TypeGuard

class Score:
    def __init__(self) -> None:
        super().__init__()
        self._value = 0
        self._count = 0

    def update(self, mean, count):
        self._value += mean * count
        self._count += count

    @property
    def mean(self):
        return self._value / self._count
    
    @property
    def count(self):
        return self._count
    
def cycle(dl):
    while True:
        for data in dl:
            yield data

T = TypeVar("T")

def exist(x: Optional[T]) -> TypeGuard[T]:
    return x is not None

def load_ckpt(model, opt, resume_ckpt, ckpt_dir, expr_name):
    if exist(resume_ckpt):
        if isinstance(resume_ckpt, int):
            ckpt = torch.load(f"{ckpt_dir}/{expr_name}/step={resume_ckpt:06}.ckpt")
        else:
            ckpt = torch.load(f"{ckpt_dir}/{expr_name}/{resume_ckpt}.ckpt")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])

def save_ckpt(model, opt, step, ckpt_dir, expr_name):
    torch.save({"model": model.state_dict(),
                "opt": opt.state_dict()},
                f"{ckpt_dir}/{expr_name}/step={step:06}.ckpt")
    torch.save({"model": model.state_dict(),
                "opt": opt.state_dict()},
                f"{ckpt_dir}/{expr_name}/last.ckpt")
    
def create_optim(params: Iterator[Parameter], opt_cfg: OptimConfig):
    return getattr(optim, opt_cfg.optim_type)(params, **opt_cfg.optim_args)