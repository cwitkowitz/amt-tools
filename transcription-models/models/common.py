# My imports
from tools.constants import *
from tools.utils import *

# Regular imports
from abc import abstractmethod
from torch import nn

import torch.nn.functional as F
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

    def pre_proc(self):
        return NotImplementedError

    @abstractmethod
    def forward(self):
        return NotImplementedError

    def post_proc(self):
        return NotImplementedError

    def run_on_batch(self, batch):
        # TODO - remove dependence on GT - put it in separate function
        feats = batch['feats']
        tabs = batch['tabs']

        tabs = tabs.transpose(-1, -2)
        tabs = tabs.transpose(-2, -3)

        feats = framify_tfr(feats, 9, 1, 4)
        feats = feats.transpose(-1, -2)
        feats = feats.transpose(-2, -3)
        feats = feats.squeeze(1)

        feats = feats.to(self.device)
        tabs = tabs.to(self.device)

        batch_size = feats.size(0)
        num_wins = feats.size(1)
        num_bins = feats.size(2)
        win_len = feats.size(3)

        feats = feats.view(batch_size * num_wins, 1, num_bins, win_len)

        out = self(feats).view(tabs.shape)

        # TODO - abstract tabs vs. pianoroll output neurons into pre-proc / post-proc (if tabs: etc.) - maybe pre-proc can be generic to the abstract class then
        preds = torch.argmax(torch.softmax(out, dim=-1), dim=-1)

        out = out.view(-1, NUM_FRETS + 2)
        tabs = tabs.view(-1, NUM_FRETS + 2)
        loss = F.cross_entropy(out, torch.argmax(tabs, dim=-1), reduction='none')
        loss = torch.sum(loss.view(-1, NUM_STRINGS), dim=-1)

        return preds, loss

    @classmethod
    def model_name(cls):
        return cls.__name__
