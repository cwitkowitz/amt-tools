# My imports
from constants import *

# Regular imports
from tqdm import tqdm
from torch import nn

import torch.nn.functional as F
import numpy as np
import librosa

class OnsetsFrames(nn.Module):
    def __init__(self, device):

        self.device = None

        self.changeDevice(device)

    def changeDevice(self, device):
        self.device = device
        self.to(self.device)

    def forward(self, batch):
        pass
