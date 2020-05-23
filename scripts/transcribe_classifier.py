"""
Run my algorithm on each file of GuitarSet and save results
"""

# My imports
from train_classifier import *
from constants import *
from utils import *

# Regular imports
from sacred import Experiment
from tqdm import tqdm

import torch.nn.functional as F
import numpy as np
import librosa
import torch
import os

ex = Experiment('Step 3: Transcribe Audio (Classifier)')

@ex.config
def config():
    # Switch for mixed hex vs. mono-channel audio
    hex_mix = False

    # Transcribe this single file if not empty
    # Example - '00_BN1-129-Eb_comp'
    single = '00_BN1-129-Eb_comp'

    # Remove this player from the split if not empty
    # Example = '00'
    # Use this attribute if a single file is not chosen
    player = '00'

    # Name of the chosen dictionary
    dict_name = '00_BN1-129-Eb_comp'

    #
    class_name = '00_BN1-129-Eb_comp'

    # GPU to use for convolutional sparse coding
    gpu_num = 0

    # Number of samples within one frame
    w_len = 512 # samples

    # Number of samples between frames
    h_len = 256 # samples

    seed = 0

    # Absolute threshold for a positive frame-wise activation
    thr = 0.5

    # Minimum number of frames a positive frame prediction must span to become a note
    # TODO - investigate this parameter
    min_note_span = 10

@ex.automain
def main(hex_mix, single, player, dict_name, class_name, gpu_num,
         w_len, h_len, seed, thr, min_note_span):

    # Obtain the track list for the chosen data partition
    track_keys = clean_track_list(GuitarSet, single, player, False)

    # Load the dictionary and activations
    elems = np.load(os.path.join(GEN_DICT_DIR, f'{dict_name}.npz'))['elems']

    M = elems.shape[0]
    elems_per = M // (NUM_FRETS + 1) // NUM_STRINGS

    acts = Activations(track_keys, dict_name, hex_mix, w_len, h_len, None, seed)

    loader = DataLoader(acts, 1, shuffle = False, num_workers = 0, drop_last = False)

    device = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'

    classifier = torch.load(os.path.join(GEN_CLASS_DIR, f'{class_name}/model.pt'))
    #classifier.eval()
    classifier.device = device
    classifier = classifier.to(classifier.device)

    for clip in tqdm(loader):
        pred, loss = classifier(clip)

        """
        x = x.view(NUM_STRINGS, NUM_FRETS + 1, 10, -1).cpu().detach().numpy()
        max_idx = np.argmax(x, axis=2)

        x = F.max_pool2d(torch.Tensor(x), (elems_per, 1), (elems_per, 1))
        x = x.view(clip['tab_onsets'].shape)
        """

        print(loss)

        onsets = np.zeros((NUM_STRINGS, NUM_NOTES, pred['onsets'].shape[-1]))

        for i in range(NUM_STRINGS):
            start_note = librosa.note_to_midi(TUNING[i]) - LOWEST_NOTE
            end_note = start_note + NUM_FRETS + 1

            onsets[i, start_note : end_note] = pred['onsets'][0, i].cpu().detach().numpy()

        onsets = onsets.transpose((0, 2, 1))

        # Apply an absolute threshold
        onsets[onsets < thr] = 0
        onsets[onsets != 0] = 1

        tabs = onsets_to_frames(onsets, elems, h_len, None)

        frames = np.max(tabs, axis=0)

        tabs = [list(extract_notes(onsets[i], tabs[i], min_note_span, w_len, h_len)) for i in range(NUM_STRINGS)]

        # Collapse the string dimension to combine predictions
        onsets = np.max(onsets, axis=0)
        pitches, ints = extract_notes(onsets, frames, min_note_span, w_len, h_len)

        # Create the activation directory if it does not already exist
        reset_generated_dir(GEN_ESTIM_DIR, [f'{dict_name}'], False)
        reset_generated_dir(os.path.join(GEN_ESTIM_DIR, f'{dict_name}'), ['frames', 'notes', 'tabs'], False)

        track_id = clip['id'][0]

        # Construct the paths for frame- and note-wise predictions
        frm_txt_path = os.path.join(GEN_ESTIM_DIR, f'{dict_name}', 'frames', f'{track_id}_class.txt')
        nte_txt_path = os.path.join(GEN_ESTIM_DIR, f'{dict_name}', 'notes', f'{track_id}_class.txt')

        # Save the predictions to file
        write_frames(frm_txt_path, w_len, h_len, frames)
        write_notes(nte_txt_path, pitches, ints)

        tabs_dir = os.path.join(GEN_ESTIM_DIR, f'{dict_name}', 'tabs', f'{track_id}')

        reset_generated_dir(tabs_dir, [], False)

        for i, s in enumerate(TUNING):
            tab_txt_path = os.path.join(tabs_dir, f'{s}_class.txt')
            pitches, ints = tabs[i]
            write_notes(tab_txt_path, pitches, ints)
