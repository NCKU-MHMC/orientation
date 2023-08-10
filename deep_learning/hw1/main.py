import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

from typing import Callable

from hw1.config import HW1Config
from hw1.data import get_data_loader
from hw1.utils import Score, cycle, load_ckpt, save_ckpt, create_optim, exist
from hw1.model import Model

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig

from pathlib import Path

from tqdm import trange, tqdm

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=HW1Config)

@torch.no_grad()
def eval(model: nn.Module,
         dl: DataLoader,
         criterion: Callable[..., Tensor],
         device: torch.device):
    model_state = model.training
    model.eval()
    for batch in dl:
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)
        logits: torch.Tensor = model(imgs)
        loss = criterion(logits, labels)
        acc = (logits.argmax(-1) == labels).float().mean()
        
        yield loss.item(), acc.item(), len(labels)
    model.train(model_state)

@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: HW1Config) -> None:
    expr_name = HydraConfig.get().job.config_name
    Path(f"{cfg.ckpt_dir}/{expr_name}").mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(f"{cfg.ckpt_dir}/{expr_name}")

    device = torch.device(f"cuda:{cfg.device_id}" if torch.cuda.is_available() else "cpu")

    model = Model().to(device)
    opt = create_optim(model.parameters(), cfg.optim)
    criterion = nn.CrossEntropyLoss()

    # load checkpoint
    load_ckpt(model, opt, cfg.resume_ckpt, cfg.ckpt_dir, expr_name)

    # training
    if cfg.mode == "train":
        train_dl = get_data_loader(cfg.data.train_paths,
                                   cfg.data.train_batch_size,
                                   cfg.data.num_workers,
                                   shuffle=True, drop_last=True)
        valid_dl = get_data_loader(cfg.data.valid_paths,
                                   cfg.data.valid_batch_size,
                                   cfg.data.num_workers,)

        score_dict = {"train/loss": 0,
                      "train/acc": 0,
                      "valid/loss": 0,
                      "valid/acc": 0}
        
        with trange(cfg.total_steps) as t:
            train_loss, train_acc = Score(), Score()
            t.set_description(expr_name)
            for batch, step in zip(cycle(train_dl), t):

                if step % cfg.save_every == 0:
                    save_ckpt(model, opt, step, cfg.ckpt_dir, expr_name)

                # validation
                if step % cfg.valid_every == 0:
                    total_valid_loss, total_valid_acc = Score(), Score()
                    for valid_loss, valid_acc, valid_bsz in eval(model, valid_dl, criterion, device,):
                        total_valid_loss.update(valid_loss, valid_bsz)
                        total_valid_acc.update(valid_acc, valid_bsz)
                        score_dict["valid/loss"] = total_valid_loss.mean
                        score_dict["valid/acc"] = total_valid_acc.mean
                        t.set_postfix(score_dict)
                        
                    writer.add_scalar("valid/loss", total_valid_loss.mean, global_step=step)
                    writer.add_scalar("valid/acc", total_valid_acc.mean, global_step=step)

                # predict and calculate loss
                imgs, labels = batch
                imgs, labels = imgs.to(device), labels.to(device)
                logits: torch.Tensor = model(imgs)
                loss: torch.Tensor = criterion(logits, labels)
                acc = (logits.argmax(-1) == labels).float().mean()

                # regularization loss
                regul_loss = 0
                if exist(cfg.optim.regul_ord):
                    for name, param in model.named_parameters():
                        if 'bias' not in name:
                            regul_loss += (param.abs() ** cfg.optim.regul_ord).sum()
                    regul_loss = cfg.optim.regul_lambda * regul_loss

                # backward and update parameters
                (loss+regul_loss).backward()
                opt.step()
                opt.zero_grad()
                
                # log training loss and acc
                writer.add_scalar("train/loss", loss, global_step=step)
                writer.add_scalar("train/acc", acc, global_step=step)
                train_loss.update(loss.item(), len(labels))
                train_acc.update(acc.item(), len(labels))
                score_dict["train/loss"] = train_loss.mean
                score_dict["train/acc"] = train_acc.mean

                if step % cfg.print_every == 0:
                    t.set_postfix(score_dict)
                    train_loss, train_acc = Score(), Score()

        save_ckpt(model, opt, cfg.total_steps, cfg.ckpt_dir, expr_name)
        # last validation
        with tqdm(total=len(valid_dl.dataset)) as pbar:
            total_valid_loss, total_valid_acc = Score(), Score()
            for valid_loss, valid_acc, valid_bsz in eval(model, valid_dl, criterion, device,):
                total_valid_loss.update(valid_loss, valid_bsz)
                total_valid_acc.update(valid_acc, valid_bsz)
                pbar.set_postfix({"valid/loss": total_valid_loss.mean,
                                  "valid/acc": total_valid_acc.mean})
                pbar.update(valid_bsz)
            writer.add_scalar("valid/loss", total_valid_loss.mean, global_step=cfg.total_steps)
            writer.add_scalar("valid/acc", total_valid_acc.mean, global_step=cfg.total_steps)

    
    # testing
    test_dl = get_data_loader(cfg.data.test_paths,
                              cfg.data.test_batch_size,
                              cfg.data.num_workers,)

    with tqdm(total=len(test_dl.dataset)) as pbar:
        total_test_loss, total_test_acc = Score(), Score()
        for test_loss, test_acc, test_bsz in eval(model, test_dl, criterion, device,):
            total_test_loss.update(test_loss, test_bsz)
            total_test_acc.update(test_acc, test_bsz)
            pbar.set_postfix({"test/loss": total_test_loss.mean,
                              "test/acc": total_test_acc.mean})
            pbar.update(test_bsz)

if __name__ == "__main__":
    main()