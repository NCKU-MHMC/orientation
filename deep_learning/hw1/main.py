import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path


from hw1.config import HW1Config
from hw1.data import get_data_loader
from hw1.utils import Score, cycle, save_ckpt
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
def eval(model, dl, criterion,):
    for batch in dl:
        imgs, labels = batch
        imgs, labels = imgs.cuda(), labels.cuda()
        logits: torch.Tensor = model(imgs)
        loss: torch.Tensor = criterion(logits, labels)
        acc = (logits.argmax(-1) == labels).float().mean()
        
        yield loss.item(), acc.item(), len(labels)

@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: HW1Config) -> None:
    expr_name = HydraConfig.get().job.config_name
    Path(f"{cfg.ckpt_dir}/{expr_name}").mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(f"{cfg.ckpt_dir}/{expr_name}")

    train_dl = get_data_loader(cfg.data.train_paths,
                            shuffle=True, drop_last=True)
    valid_dl = get_data_loader(cfg.data.valid_paths)
    test_dl = get_data_loader(cfg.data.test_paths)

    model = Model().cuda()
    opt = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    if cfg.resume_iter != None:
        if cfg.resume_iter=="last":
            ckpt = torch.load(f"{cfg.ckpt_dir}/{expr_name}/last.ckpt")
        else:
            ckpt = torch.load(f"{cfg.ckpt_dir}/{expr_name}/step={cfg.resume_iter}.ckpt")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])

    score_dict = {"train/loss": 0,
                  "train/acc": 0,
                  "valid/loss": 0,
                  "valid/acc": 0}
    
    with trange(cfg.total_iters) as t:
        train_loss, train_acc = Score(), Score()
        for batch, batch_idx in zip(cycle(train_dl), t):

            imgs, labels = batch
            imgs, labels = imgs.cuda(), labels.cuda()
            logits: torch.Tensor = model(imgs)
            loss: torch.Tensor = criterion(logits, labels)
            acc = (logits.argmax(-1) == labels).float().mean()

            loss.backward()
            opt.step()
            opt.zero_grad()
            
            writer.add_scalar("train/loss", loss, global_step=batch_idx)
            writer.add_scalar("train/acc", acc, global_step=batch_idx)
            train_loss.update(loss.item(), len(labels))
            train_acc.update(acc.item(), len(labels))
            score_dict["train/loss"] = train_loss.mean
            score_dict["train/acc"] = train_acc.mean

            if batch_idx % cfg.save_every == 0:
                save_ckpt(model, opt, batch_idx, cfg.ckpt_dir, expr_name)

            if batch_idx % cfg.print_every == 0:
                t.set_postfix(score_dict)

            if batch_idx % cfg.valid_every == 0:
                total_valid_loss, total_valid_acc = Score(), Score()
                for valid_loss, valid_acc, valid_bsz in eval(model, valid_dl, criterion,):
                    total_valid_loss.update(valid_loss, valid_bsz)
                    total_valid_acc.update(valid_acc, valid_bsz)
                    score_dict["valid/loss"] = total_valid_loss.mean
                    score_dict["valid/acc"] = total_valid_acc.mean
                    t.set_postfix(score_dict)
                    
                writer.add_scalar("valid/loss", total_valid_loss.mean, global_step=batch_idx)
                writer.add_scalar("valid/acc", total_valid_acc.mean, global_step=batch_idx)

    save_ckpt(model, opt, batch_idx, cfg.ckpt_dir, expr_name)
    
    with tqdm(total=len(test_dl.dataset)) as pbar:
        total_test_loss, total_test_acc = Score(), Score()
        for test_loss, test_acc, test_bsz in eval(model, test_dl, criterion,):
            total_test_loss.update(test_loss, test_bsz)
            total_test_acc.update(test_acc, test_bsz)
            pbar.set_postfix({"test/loss": total_test_loss.mean,
                              "test/acc": total_test_acc.mean})
            pbar.update(test_bsz)

if __name__ == "__main__":
    main()