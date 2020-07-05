"""
TODO
"""

# My imports
from auxiliary.constants import *
from auxiliary.datasets import *
from auxiliary.dataproc import *
from auxiliary.models import *
from auxiliary.utils import *

# Regular imports
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from sacred import Experiment
from tqdm import tqdm
from torch import nn

import torch.nn.functional as F
import numpy as np
import librosa
import random
import torch
import jams
import os

ex = Experiment('Train Classifier')

@ex.config
def config():
    splits = ['01', '02', '03', '04', '05']

    # Number of samples between frames
    hop_length = 512 # samples

    # GPU to use for convolutional sparse coding
    gpu_num = 0

    iters = 2000

    batch_size = 128

    l_rate = 1e0

@ex.automain
def train_classifier(splits, hop_length, gpu_num, iters, batch_size, l_rate):
    seed_everything(SEED)

    class_dir = f'TabCNN'

    class_dir = os.path.join(GEN_CLASS_DIR, class_dir)
    out_path = os.path.join(class_dir, 'model.pt')

    data_proc = CQT(hop_length, None, 192, 24)

    gset_fold = GuitarSet(None, splits, hop_length, data_proc, 10)

    loader = DataLoader(gset_fold, batch_size, shuffle=True, num_workers=16, drop_last=True)

    device = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'

    classifier = TabCNN(device)
    classifier.train()

    # TODO - adam? - pair this with the model?
    optimizer = torch.optim.Adadelta(classifier.parameters(), l_rate)

    os.makedirs(class_dir, exist_ok=True)
    writer = SummaryWriter(class_dir)

    # TODO - explicit epochs?
    for i in tqdm(range(iters)):
        for batch in loader:
            optimizer.zero_grad()
            preds, loss = classifier.run_on_batch(batch)
            loss = torch.mean(loss)
            writer.add_scalar(f'train_loss', torch.mean(loss), global_step=i)
            loss.backward()
            optimizer.step()

    if os.path.exists(out_path):
        os.remove(out_path)

    torch.save(classifier, out_path)
