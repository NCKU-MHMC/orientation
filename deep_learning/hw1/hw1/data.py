import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms as T

from einops import rearrange

from typing import List, Optional, Union, Callable
from pathlib import Path
import pickle

from hw1.utils import default_lazy, Lambda

def get_data_loader(paths: List[Union[str, Path]],
                    batch_size = 64,
                    num_workers = 0,
                    augment = False,
                    shuffle = False,
                    drop_last = False):
    transform = T.RandAugment() if augment else None
    dataset = ImageNet32Dataset(paths, transform)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                      shuffle=shuffle, drop_last=drop_last, collate_fn=dataset.seqCollate)

class ImageNet32Dataset(Dataset):
    def __init__(self,
                 paths: List[Union[str, Path]],
                 transform: Optional[Callable[[Tensor], Tensor]] = None):
        super().__init__()
        imgs_ls = []
        labels = []
        for p in paths:
            with open(p, 'rb') as fo:
                data = pickle.load(fo)
                x = torch.from_numpy(data['data'],)
                x = rearrange(x, "b (c h w) -> b c h w",
                              c=3, h=32, w=32)
                
                imgs_ls.append(x)
                labels += [label-1 for label in data['labels']]

        self.imgs = torch.cat(imgs_ls)
        self.labels = labels
        self.transforms = nn.Sequential(default_lazy(transform, nn.Identity),
                                        Lambda(lambda x: x/255),
                                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.imgs[i], self.labels[i]

    def seqCollate(self, batch):
        imgs, labels = list(zip(*batch))
        imgs = [self.transforms(img) for img in imgs]
        imgs, labels = torch.stack(imgs), torch.tensor(labels)
        return imgs, labels