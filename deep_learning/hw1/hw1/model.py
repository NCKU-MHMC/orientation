from torch import nn
from einops import rearrange

class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        net = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                            nn.Conv2d(32, 64, 3, 2, 1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.Conv2d(64, 128, 3, 2, 1),
                            nn.BatchNorm2d(128),
                            nn.ReLU(),
                            nn.AdaptiveAvgPool2d((1,1)),
                            nn.Conv2d(128, 256, 1,),
                            nn.ReLU(),
                            nn.Conv2d(256, 1000, 1,),)
        self.net = net
        
    def forward(self, imgs):
        logits = rearrange(self.net(imgs), "b c 1 1 -> b c")
        return logits