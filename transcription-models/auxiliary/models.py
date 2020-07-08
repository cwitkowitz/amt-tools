# My imports
from .constants import *
from .utils import *

# Regular imports
from abc import abstractmethod
from torch import nn

import torch.nn.functional as F
import torch
import math

# TODO - separate modules for output classifier? or ways of transcribing notes?


class TranscriptionModel(nn.Module):
    def __init__(self, dim_in, dim_out, model_complexity=1, device='cpu'):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.model_complexity = model_complexity

        self.device = None
        self.change_device(device)

    def change_device(self, device):
        self.device = device
        self.to(self.device)

    @abstractmethod
    def forward(self):
        return NotImplementedError

    @abstractmethod
    def pre_proc(self):
        return NotImplementedError

    @abstractmethod
    def post_proc(self):
        return NotImplementedError

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

class TabCNN(TranscriptionModel):
    def __init__(self, dim_in, dim_out, model_complexity=1, device='cpu'):
        super().__init__(dim_in, dim_out, model_complexity, device)

        # Number of frames required for a prediction
        sample_width = 9

        # Number of filters for each stage
        nf1 = 32 * self.model_complexity
        nf2 = 64 * self.model_complexity
        nf3 = nf2

        # Kernel size for each stage
        ks1 = (3, 3)
        ks2 = ks1
        ks3 = ks1

        # Reduction size for each stage
        rd1 = (2, 2)

        # Dropout percentages for each stage
        dp1 = 0.25
        dp2 = 0.50

        # Number of neurons for each fully-connected stage
        nn1 = 128
        nn2 = dim_out

        # Stage 1 convolution
        self.cn1 = nn.Conv2d(1, nf1, ks1)
        # Stage 2 convolution
        self.cn2 = nn.Conv2d(nf1, nf2, ks2)
        # Stage 3 convolution
        self.cn3 = nn.Conv2d(nf2, nf3, ks3)
        # Stage 1 reduction
        self.mp1 = nn.MaxPool2d(rd1)
        # Stage 1 dropout
        self.dp1 = nn.Dropout(dp1)

        feat_map_height = (dim_in - 6) / 2
        feat_map_width = (sample_width - 6) / 2
        self.feat_map_size = nf3 * feat_map_height * feat_map_width

        # Stage 1 fully-connected
        self.fc1 = nn.Linear(self.feat_map_size, nn1)
        # Stage 2 dropout
        self.dp2 = nn.Dropout(dp2)
        # Stage 2 fully-connected
        self.fc2 = nn.Linear(nn1, nn2)

    def forward(self, feats):
        # Stage 1 convolution
        x = F.relu(self.cn1(feats))
        # Stage 2 convolution
        x = F.relu(self.cn2(x))
        # Stage 3 convolution
        x = F.relu(self.cn3(x))
        # Stage 1 reduction
        x = self.mp1(x).flatten()
        # Stage 1 dropout
        x = self.dp1(x)
        # Stage 1 fully-connected
        x = x.view(-1, self.feat_map_size)
        x = F.relu(self.fc1(x))
        # Stage 2 dropout
        x = self.dp2(x)
        # Stage 2 fully-connected
        out = self.fc2(x)

        return out

class OnsetsFrames(TranscriptionModel):
    # TODO - separate acoustic model from mlm
    def __init__(self, device):

        self.device = None

        self.changeDevice(device)

    def changeDevice(self, device):
        self.device = device
        self.to(self.device)

    def forward(self, batch):
        pass
