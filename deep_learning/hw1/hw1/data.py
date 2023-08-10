import torch
from torch.utils.data import Dataset, DataLoader

from einops import rearrange

from typing import List, Union
from pathlib import Path
import pickle

def get_data_loader(paths: List[Union[str, Path]],
                    batch_size = 64,
                    num_workers = 0,
                    shuffle = False,
                    drop_last = False):
    dataset = ImageNet32Dataset(paths)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                      shuffle=shuffle, drop_last=drop_last)

class ImageNet32Dataset(Dataset):
    def __init__(self,
                 paths: List[Path]):
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

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        return self.imgs[i]/255, self.labels[i]

    def seqCollate(self, batch):
        x, y = list(zip(batch))
        return torch.stack(x), torch.tensor(y)