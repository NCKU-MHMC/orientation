import torch

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

def save_ckpt(model, opt, batch_idx, ckpt_dir, expr_name):
    torch.save({"model": model.state_dict(),
                "opt": opt.state_dict()},
                f"{ckpt_dir}/{expr_name}/step={batch_idx:06}.ckpt")
    torch.save({"model": model.state_dict(),
                "opt": opt.state_dict()},
                f"{ckpt_dir}/{expr_name}/last.ckpt")