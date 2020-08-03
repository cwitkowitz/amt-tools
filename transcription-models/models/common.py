# My imports
from tools.constants import *
from tools.utils import *

# Regular imports
from abc import abstractmethod
from torch import nn

import torch


class TranscriptionModel(nn.Module):
    def __init__(self, dim_in, dim_out, model_complexity=1, device='cpu'):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.model_complexity = model_complexity
        self.device = device

    def change_device(self, device=None):
        if device is None:
            device = self.device

        device = torch.device(f'cuda:{device}'
                              if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.to(self.device)

    @abstractmethod
    def pre_proc(self, batch):
        return NotImplementedError

    @abstractmethod
    def forward(self, feats):
        return NotImplementedError

    @abstractmethod
    def post_proc(self, batch):
        return NotImplementedError

    def run_on_batch(self, batch):
        batch = self.pre_proc(batch)

        batch['out'] = self(batch['feats'])

        preds, loss = self.post_proc(batch)

        return preds, loss

    @classmethod
    def model_name(cls):
        return cls.__name__
