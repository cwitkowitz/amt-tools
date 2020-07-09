"""
TODO
"""

# My imports
from tools.constants import *
from tools.datasets import *
from tools.dataproc import *
from tools.models import *
from tools.utils import *

# Regular imports
from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch
import os

def train(classifier, loader, optimizer, iterations, checkpoints, log_dir):
    # TODO - validation loader if we want to validate
    # TODO - resume mechanism
    # TODO - scheduler

    writer = SummaryWriter(log_dir)

    for i in tqdm(range(iterations)):
        for batch in loader:
            optimizer.zero_grad()
            _, loss = classifier.run_on_batch(batch)
            loss = torch.mean(loss)
            writer.add_scalar(f'train_loss', torch.mean(loss), global_step=i)
            loss.backward()
            optimizer.step()

        if (i + 1) % checkpoints == 0:
            torch.save(classifier, os.path.join(log_dir, f'classifier-{i + 1}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(log_dir, f'opt-state-{i + 1}.pt'))

    torch.save(classifier, os.path.join(log_dir, f'classifier-{i + 1}.pt'))
    torch.save(optimizer.state_dict(), os.path.join(log_dir, f'opt-state-{i + 1}.pt'))

    return classifier
