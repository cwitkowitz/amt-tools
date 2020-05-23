"""
TODO
"""

# My imports
from constants import *
from utils import *

# Regular imports
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from sacred import Experiment
from tqdm import tqdm
from torch import nn

import matplotlib.pyplot as plt
import torch.nn.functional as F
import soundfile as sf
import numpy as np
import mirdata
import librosa
import random
import torch
import os

# TODO - seed this S

ex = Experiment('Step 2.5: Train Classifier')

class GuitarSet(Dataset):
    def __init__(self, track_ids, win_len, hop_len, frm_len, seed):
        self.track_ids = track_ids

        self.win_len = win_len
        self.hop_len = hop_len
        self.frm_len = frm_len
        self.random = np.random.RandomState(seed)

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, index):
        track_name = self.track_ids[index]
        track = GuitarSetHandle[track_name]

        save_path = os.path.join(GEN_GT_DIR, f'{track_name}.npz')

        data = {}

        if os.path.exists(save_path):
            clip = np.load(save_path)

            cqt = clip['cqt']
            tabs = clip['tabs']
        else:
            audio, fs = track.audio_mic

            assert fs == SAMPLE_RATE

            notes = track.notes  # Dictionary of notes for this track

            cqt = abs(librosa.cqt(audio, fs, self.hop_len, n_bins=192, bins_per_octave=24))

            num_frames = cqt.shape[-1]

            tabs = np.zeros((NUM_FRETS + 2, num_frames))

            for i, s_key in enumerate(notes.keys()):
                s_data = notes[s_key]
                onset = ((s_data.start_times * SAMPLE_RATE) // self.hop_len).astype('uint32')
                offset = ((s_data.end_times * SAMPLE_RATE) // self.hop_len).astype('uint32')

                fret = (np.array(s_data.notes) - librosa.note_to_midi(TUNING[i])).astype('uint32')

                for n in range(len(fret)):
                    tabs[fret[n], onset[n]:offset[n]] = 1

                tabs[-1, np.sum(tabs, axis=0) == 0] = 1

            np.savez(save_path, cqt=cqt, tabs=tabs)

        if self.frame_len is not None:
            step_begin = self.random.randint(num_frames - self.frame_len)
            step_end = step_begin + self.frame_len

            data['cqt'] = cqt[:, step_begin:step_end]
            data['tabs'] = tabs[:, step_begin:step_end]
        else:
            data['cqt'] = cqt
            data['tabs'] = tabs

        data['id'] = track_name
        data['cqt'] = torch.from_numpy(data['cqt']).float()
        data['tabs'] = torch.from_numpy(data['tabs']).float()

        return data

class TabCNN(nn.Module):
    def __init__(self, device, elems_per):
        super().__init__()

        self.device = device

        num_filters = 1

        num_elems = elems_per * NUM_GROUPS

        self.bn1 = nn.BatchNorm2d(num_filters)
        self.cn1 = nn.Conv2d(1, num_filters, (1, 512), (1, 256), (0, 0))
        self.cn2 = nn.Conv2d(num_filters, 1, (1, 7), 1, (0, 3))
        self.pl1 = nn.LPPool2d(1, (elems_per, 1), (elems_per, 1))

        """
        self.cn1 = nn.Conv2d(1, num_filters, (elems_per, 9), (elems_per, 1), (0, 4))
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.rl1 = nn.ReLU()

        self.pl1 = nn.MaxPool2d((elems_per, 1), (elems_per, 1))
        self.dp1 = nn.Dropout()

        self.fc_in = num_filters * NUM_STRINGS * (NUM_FRETS + 1)

        self.fco = nn.Linear(self.fc_in, num_groups)
        self.dp2 = nn.Dropout()
        self.gro = nn.GRU(num_groups, num_groups, batch_first=True)

        self.fcf = nn.Linear(self.fc_in, num_groups)
        self.dp3 = nn.Dropout()
        self.grf = nn.GRU(num_groups, num_groups, batch_first=True)
        """

        self.to(device)

    def forward(self, batch):
        acts = batch['samples'].to(self.device)
        onsets = batch['tab_onsets'].to(self.device)
        frames = batch['tab_frames'].to(self.device)

        """
        #x = self.rl1(self.bn1(self.cn1(acts.unsqueeze(1))))
        x = self.rl1(self.cn1(acts.unsqueeze(1)))
        #x = self.dp1(self.pl1(x))

        bs, nflts, nntes, nfrms = x.size()

        onsets = onsets.view(bs, num_groups, -1)
        frames = frames.view(bs, num_groups, -1)

        x = x.permute(0, 3, 1, 2).view(bs, nfrms, nflts * nntes)

        x_o, _ = self.gro(self.dp2(self.fco(x)), torch.randn(1, 1, num_groups).to(self.device))
        x_f, _ = self.grf(self.dp3(self.fcf(x)), torch.randn(1, 1, num_groups).to(self.device))

        x_o = x_o.permute(0, 2, 1)
        x_f = x_f.permute(0, 2, 1)

        loss = {'onset': F.binary_cross_entropy_with_logits(x_o, onsets),
                'frame': F.binary_cross_entropy_with_logits(x_f, frames)}

        preds = {'onsets': torch.sigmoid(x_o), 'frames': torch.sigmoid(x_f)}
        """

        x = self.bn1(self.cn1(acts.unsqueeze(1)))
        #x = self.cn1(acts.unsqueeze(1))
        #x = torch.relu(self.cn1(self.bn1(acts.unsqueeze(1))))
        #x = torch.relu(self.cn1(acts.unsqueeze(1)))
        #x = self.cn2(torch.relu(self.cn1(self.bn1(acts.unsqueeze(1)))))
        #x = self.pl1(self.cn2(self.cn1(acts.unsqueeze(1))))
        #x = F.relu(self.cn1(acts.unsqueeze(1)))
        x = self.pl1(torch.max(x, dim = 1)[0])

        x = x.view(batch['tab_onsets'].shape)

        loss = {'onset': F.binary_cross_entropy_with_logits(x, batch['tab_onsets'].to(self.device))}
        preds = {'onsets': torch.sigmoid(x)}

        return preds, loss

@ex.config
def config():
    # Use this single file if not empty
    # Example - '00_BN1-129-Eb_comp'
    single = '00_BN1-129-Eb_comp'

    # Remove this player from the split if not empty
    # Example = '00'
    # Use this attribute if a single file is not chosen
    player = '00'

    # Name of the chosen dictionary
    dict_name = '00_BN1-129-Eb_comp'

    win_len = 512 # samples

    # Number of samples between frames
    hop_len = 512 # samples

    # GPU to use for convolutional sparse coding
    gpu_num = 0

    frm_len = 9

    iterations = 5000

    batch_size = 1

    l_rate = 1e-2

    seed = 0

@ex.automain
def main(single, player, dict_name, win_len, hop_len, gpu_num, frm_len, iterations, batch_size, l_rate, seed):
    # Create the activation directory if it does not already exist

    # Path for saving the dictionary
    if single == '':
        class_dir = f'excl_{player}'
    else:
        class_dir = f'{single}'

    reset_generated_dir(GEN_CLASS_DIR, [class_dir], False)
    reset_generated_dir(GEN_GT_DIR, [], True)

    class_dir = os.path.join(GEN_CLASS_DIR, class_dir)
    out_path = os.path.join(class_dir, 'model.pt')

    os.makedirs(class_dir, exist_ok=True)
    writer = SummaryWriter(class_dir)

    # Obtain the track list for the chosen data partition
    track_keys = clean_track_list(GuitarSet, single, player, False)

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    tabs = GuitarSet(track_keys, win_len, hop_len, frm_len, seed)

    loader = DataLoader(tabs, batch_size, shuffle=True, num_workers=0, drop_last=True)

    device = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'

    classifier = TabCNN(device)
    classifier.train()

    optimizer = torch.optim.Adam(classifier.parameters(), l_rate)
    scheduler = StepLR(optimizer, step_size=iterations/2, gamma=0.98)

    for i in tqdm(range(iterations)):
        for batch in loader:
            preds, loss = classifier(batch)
            writer.add_scalar(f'train_loss', sum(loss.values()), global_step = i)
            #print(loss)

            optimizer.zero_grad()
            torch.mean(sum(loss.values())).backward()
            optimizer.step()
            scheduler.step()

    if os.path.exists(out_path):
        os.remove(out_path)

    torch.save(classifier, out_path)
