"""
Run my algorithm on each file of GuitarSet and save results
"""

# My imports
from auxiliary.constants import *
from auxiliary.datasets import *
from auxiliary.dataproc import *
from auxiliary.models import *
from auxiliary.utils import *

# Regular imports
from torch.utils.data import Dataset, DataLoader
from sacred import Experiment
from tqdm import tqdm

import numpy as np
import librosa
import torch
import os

ex = Experiment('Transcribe Audio')

@ex.config
def config():
    splits = ['00']

    # Number of samples between frames
    hop_length = 512 # samples

    # GPU to use for convolutional sparse coding
    gpu_num = 0

    # Minimum number of frames a positive frame prediction must span to become a note
    min_note_span = 5

@ex.automain
def transcribe_classifier(splits, hop_length, gpu_num, min_note_span):
    data_proc = CQT(hop_length, None, 192, 24)

    gset_test = GuitarSet(None, splits, hop_length, data_proc, None)

    loader = DataLoader(gset_test, 1, shuffle=False, num_workers=0, drop_last=False)
    device = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'

    class_dir = f'TabCNN'
    classifier = torch.load(os.path.join(GEN_CLASS_DIR, f'{class_dir}/model.pt'))
    classifier.changeDevice(device)
    classifier.eval()

    for track in tqdm(loader):
        track_id = track['track'][0]

        tabs, loss = classifier.run_on_batch(track)

        tabs = tabs.squeeze().cpu().detach().numpy().T
        loss = loss.cpu().detach().numpy()

        string_pianoroll = tabs_to_multi_pianoroll(tabs)

        string_notes = [extract_notes(string_pianoroll[i], hop_length, min_note_span) for i in range(NUM_STRINGS)]

        pianoroll = tabs_to_pianoroll(tabs).T

        all_pitches, all_ints = [], []

        os.makedirs(os.path.join(GEN_ESTIM_DIR, 'frames'), exist_ok=True)
        os.makedirs(os.path.join(GEN_ESTIM_DIR, 'notes'), exist_ok=True)
        os.makedirs(os.path.join(GEN_ESTIM_DIR, 'tabs'), exist_ok=True)
        # Create the activation directory if it does not already exist

        tabs_dir = os.path.join(GEN_ESTIM_DIR, 'tabs', f'{track_id}')
        os.makedirs(os.path.join(tabs_dir), exist_ok=True)

        for i, s in enumerate(TUNING):
            tab_txt_path = os.path.join(tabs_dir, f'{s}.txt')
            pitches, ints = string_notes[i]
            all_pitches += list(pitches)
            all_ints += list(ints)
            write_notes(tab_txt_path, pitches, ints)

        all_pitches, all_ints = np.array(all_pitches), np.array(all_ints)

        # Construct the paths for frame- and note-wise predictions
        frm_txt_path = os.path.join(GEN_ESTIM_DIR, 'frames', f'{track_id}.txt')
        nte_txt_path = os.path.join(GEN_ESTIM_DIR, 'notes', f'{track_id}.txt')

        # Save the predictions to file
        write_frames(frm_txt_path, hop_length, pianoroll)
        write_notes(nte_txt_path, all_pitches, all_ints)
