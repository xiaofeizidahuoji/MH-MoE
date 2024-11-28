import torch
import torch.nn as nn
from .tsm import *
    
class Adapter(nn.Module):
    def __init__(self, args, output_features=None):
        super().__init__()
        in_features = args.input_dim
        hidden_features = args.hidden_dim
        if output_features == None:
            output_features = in_features
        drop = args.drop
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, output_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x1):
        x = self.fc1(x1)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x + x1      # residual
        return x