from torch import nn, Tensor
from torch.nn import functional as F
from typing import List
from jaxtyping import Float
from hw1.resnet import resnet18

class Classifier(nn.Module):
    def __init__(self,
                 feature_dim: int = 512,
                 num_classes: int = 1000,
                 hidden_dims: List[int] = [1024, 1024],
                 drop_rate: float = 0.1,
                 ) -> None:
        super().__init__()

        self.drop_rate = drop_rate

        in_dims = [feature_dim] + hidden_dims
        out_dims = hidden_dims + [num_classes]
        self.layers = nn.ModuleList()
        for d_in, d_out in zip(in_dims, out_dims):
            self.layers.append(nn.Linear(d_in, d_out))
    
    def forward(self, features:Float[Tensor, "b c h w"]):
        h = features.mean((2,3))
        for layer in self.layers[:1]:
            h = layer(h)
            h = F.relu(h)
            h = F.dropout(h, self.drop_rate, self.training)

        logits = self.layers[-1](h)
        return logits

class Model(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        
        self.feature_extractor = resnet18()
        self.classifier = Classifier()
        
    def forward(self, imgs):
        h = self.feature_extractor(imgs)
        logits = self.classifier(h)
        return logits


