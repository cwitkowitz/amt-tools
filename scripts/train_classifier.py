"""
TODO
"""

# My imports
from constants import *
from utils import *

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

# TODO - seed this S

ex = Experiment('Train Classifier')

@ex.config
def config():
    # Use this single file if not empty
    # Example - '00_BN1-129-Eb_comp'
    single = ''#'00_BN1-129-Eb_comp'

    # Remove this player from the split if not empty
    # Example = '00'
    # Use this attribute if a single file is not chosen
    player = '01'

    win_len = 512 # samples

    # Number of samples between frames
    hop_len = 512 # samples

    # GPU to use for convolutional sparse coding
    gpu_num = 0

    iters = 8000

    batch_size = 300

    l_rate = 1e0

    seed = 0

class GuitarSet(Dataset):
    def __init__(self, track_ids, win_len, hop_len, mode, seed):
        self.track_ids = track_ids

        self.win_len = win_len
        self.hop_len = hop_len
        self.mode = mode
        self.random = np.random.RandomState(seed)

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, index):
        track_name = self.track_ids[index]
        track = GuitarSetHandle[track_name]

        save_path = os.path.join(GEN_GT_DIR, f'{track_name}.npz')

        data = {}

        cnt_wn = 9

        if os.path.exists(save_path):
            clip = np.load(save_path)

            cqt = clip['cqt']
            tabs = clip['tabs']
        else:
            audio, fs = track.audio_mic

            audio = librosa.util.normalize(audio)

            audio = librosa.resample(audio, fs, SAMPLE_RATE)

            notes = track.notes  # Dictionary of notes for this track

            cqt = abs(librosa.cqt(audio, fs, self.hop_len, n_bins=192, bins_per_octave=24))

            num_bins = cqt.shape[0]
            num_frames = cqt.shape[-1]

            pad = cnt_wn // 2

            cqt = np.concatenate((np.zeros((num_bins, pad)),
                                  cqt, np.zeros((num_bins, pad))), axis=-1)

            cqt = np.expand_dims(cqt, axis=0)

            tabs = np.zeros((NUM_STRINGS, NUM_FRETS + 2, num_frames))

            """
            for i, s_key in enumerate(notes.keys()):
                s_data = notes[s_key]
                onset = np.round((s_data.start_times * SAMPLE_RATE) // self.hop_len).astype('uint32')
                offset = np.round((s_data.end_times * SAMPLE_RATE) // self.hop_len).astype('uint32')

                fret = np.round(np.array(s_data.notes) - librosa.note_to_midi(TUNING[i])).astype('uint32')

                for n in range(len(fret)):
                    tabs[i, fret[n], onset[n]:offset[n]] = 1

                tabs[i, -1, np.sum(tabs[i], axis=0) == 0] = 1
            """

            jam = jams.load(track.jams_path)
            frame_indices = range(num_frames)
            t_ref = librosa.frames_to_time(frame_indices, SAMPLE_RATE, self.hop_len)

            for s in range(NUM_STRINGS):
                anno = jam.annotations['note_midi'][s]
                pitch = anno.to_samples(t_ref)
                silent = [pitch[i] == [] for i in range(len(pitch))]
                tabs[s, -1, silent] = 1
                for i in range(len(pitch)):
                    if silent[i]:
                        pitch[i] = [0]
                pitch = np.array(pitch).squeeze()
                midi_pitches = (np.round(pitch[pitch != 0] - librosa.note_to_midi(TUNING[s]))).astype('uint32')
                tabs[s, midi_pitches, pitch != 0] = 1

            np.savez(save_path, cqt=cqt, tabs=tabs)

        num_frames = cqt.shape[-1] - cnt_wn + 1

        if self.mode == 'train':
            step_begin = self.random.randint(num_frames)
            step_end = step_begin + cnt_wn

            data['cqt'] = cqt[:, :, step_begin:step_end]
            data['tabs'] = tabs[:, :, step_begin]
        else:
            # Batch size must be 1
            cqt = np.concatenate([cqt[:, :, i : i + cnt_wn] for i in range(num_frames)], axis=0)
            data['cqt'] = np.expand_dims(cqt, axis=1)
            data['tabs'] = np.transpose(tabs, (2, 0, 1))

        data['id'] = track_name
        data['cqt'] = torch.from_numpy(data['cqt']).float()
        data['tabs'] = torch.from_numpy(data['tabs']).long()

        return data

class TabCNN(nn.Module):
    def __init__(self, device):
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

    def forward(self, batch):
        cqt = batch['cqt'].to(self.device)
        tabs = batch['tabs'].to(self.device)

        x = F.relu(self.cn1(cqt))
        x = F.relu(self.cn2(x))
        x = F.relu(self.cn3(x))
        x = self.mxp(x).flatten().view(-1, 5952)
        x = self.dp1(x)
        x = F.relu(self.fc1(x))
        x = self.dp2(x)
        out = self.fc2(x).view(tabs.shape)

        preds = torch.argmax(torch.softmax(out, dim=-1), dim=-1)

        out = out.view(-1, NUM_FRETS + 2)
        tabs = tabs.view(-1, NUM_FRETS + 2)
        loss = F.cross_entropy(out, torch.argmax(tabs, dim=-1), reduction='none')
        loss = torch.sum(loss.view(-1, NUM_STRINGS), dim=-1)

        return preds, loss

@ex.automain
def main(single, player, win_len, hop_len, gpu_num, iters, batch_size, l_rate, seed):
    # Create the activation directory if it does not already exist

    # Path for saving the dictionary
    if single == '':
        class_dir = f'excl_{player}'
    else:
        class_dir = f'{single}'

    reset_generated_dir(GEN_CLASS_DIR, [class_dir], False)
    reset_generated_dir(GEN_GT_DIR, [], False)

    class_dir = os.path.join(GEN_CLASS_DIR, class_dir)
    out_path = os.path.join(class_dir, 'model.pt')

    os.makedirs(class_dir, exist_ok=True)
    writer = SummaryWriter(class_dir)

    # Obtain the track list for the chosen data partition
    track_keys = clean_track_list(GuitarSetHandle, single, player, True)

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    train_tabs = GuitarSet(track_keys, win_len, hop_len, 'train', seed)

    loader = DataLoader(train_tabs, batch_size, shuffle=True, num_workers=16, drop_last=False)

    device = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'

    classifier = TabCNN(device)
    classifier.train()

    optimizer = torch.optim.Adadelta(classifier.parameters(), l_rate)

    # TODO - explicit epochs
    for i in tqdm(range(iters)):
        for batch in loader:
            optimizer.zero_grad()
            preds, loss = classifier(batch)
            writer.add_scalar(f'train_loss', torch.mean(loss), global_step=i)
            torch.mean(loss).backward()
            optimizer.step()

    if os.path.exists(out_path):
        os.remove(out_path)

    torch.save(classifier, out_path)
