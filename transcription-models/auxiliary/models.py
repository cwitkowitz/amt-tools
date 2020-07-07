# My imports
from .constants import *
from .utils import *

# Regular imports
from torch import nn

import torch.nn.functional as F
import torch

# TODO - separate modules for output classifier? or ways of transcribing notes?

class TabCNN(nn.Module):
    def __init__(self, device):
        # TODO - parameterize number of input/output features and complexity
        super().__init__()

        self.device = None

        self.cn1 = nn.Conv2d(1, 32, 3)
        self.cn2 = nn.Conv2d(32, 64, 3)
        self.cn3 = nn.Conv2d(64, 64, 3)
        self.mxp = nn.MaxPool2d(2)
        self.dp1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(5952, 128)
        self.dp2 = nn.Dropout(0.50)
        self.fc2 = nn.Linear(128, 126)

        self.changeDevice(device)

    def changeDevice(self, device):
        self.device = device
        self.to(self.device)

    def forward(self, feats):
        x = F.relu(self.cn1(feats))
        x = F.relu(self.cn2(x))
        x = F.relu(self.cn3(x))
        x = self.mxp(x).flatten().view(-1, 5952)
        x = self.dp1(x)
        x = F.relu(self.fc1(x))
        x = self.dp2(x)
        out = self.fc2(x)

        return out

    def run_on_batch(self, batch):
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

        preds = torch.argmax(torch.softmax(out, dim=-1), dim=-1)

        out = out.view(-1, NUM_FRETS + 2)
        tabs = tabs.view(-1, NUM_FRETS + 2)
        loss = F.cross_entropy(out, torch.argmax(tabs, dim=-1), reduction='none')
        loss = torch.sum(loss.view(-1, NUM_STRINGS), dim=-1)

        return preds, loss

class OnsetsFrames(nn.Module):
    # TODO - separate acoustic model from mlm
    def __init__(self, device):

        self.device = None

        self.changeDevice(device)

    def changeDevice(self, device):
        self.device = device
        self.to(self.device)

    def forward(self, batch):
        pass
