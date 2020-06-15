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

import numpy as np
import librosa
import torch
import os

ex = Experiment('Transcribe Audio')

@ex.config
def config():

    # Transcribe this single file if not empty
    # Example - '00_BN1-129-Eb_comp'
    single = ''#'00_BN1-129-Eb_comp'

    # Remove this player from the split if not empty
    # Example = '00'
    # Use this attribute if a single file is not chosen
    player = '03'

    #
    class_dir = 'excl_03'#'00_BN1-129-Eb_comp'

    win_len = 512 # samples

    # Number of samples between frames
    hop_len = 512 # samples

    # GPU to use for convolutional sparse coding
    gpu_num = 0

    # Minimum number of frames a positive frame prediction must span to become a note
    min_note_span = 5

def extract_notes(frames, win_len, hop_len, min_note_span):
    # Create empty lists for note pitches and their time intervals
    pitches, ints = [], []

    onsets = np.concatenate([frames[:, :1], frames[:, 1:] - frames[:, :-1]], axis=1) == 1

    # Find the nonzero indices
    nonzeros = onsets.nonzero()
    for i in range(len(nonzeros[0])):
        # Get the frame and pitch index
        pitch, frame = nonzeros[0][i], nonzeros[1][i]

        # Mark onset and start offset counter
        onset, offset = frame, frame

        # Increment the offset counter until the pitch activation
        # turns negative or until the last frame is reached
        while frames[pitch, offset]:
            if onset == offset and np.sum(onsets[:, max(0, onset - int(0.10 * SAMPLE_RATE // hop_len)) : onset]) > 0:
                break
            offset += 1
            if offset == frames.shape[1]:
                break

        # Make sure the note duration exceeds a minimum frame length
        if offset >= onset + min_note_span:
            # Determine the absolute frequency
            freq = librosa.midi_to_hz(pitch + LOWEST_NOTE)

            # Add the frequency to the list
            pitches.append(freq)

            # TODO - can probs utilize librosa here
            # Determine the time where the onset and offset occur
            onset, offset = onset * hop_len / SAMPLE_RATE, offset * hop_len / SAMPLE_RATE

            # Add half of the window time for frame-centered predictions
            bias = (0.5 * win_len / SAMPLE_RATE)
            onset, offset = onset + bias, offset + bias

            # Add the interval to the list
            ints.append([onset, offset])

    # Convert the lists to numpy arrays
    pitches, intervals = np.array(pitches), np.array(ints)

    return pitches, intervals

@ex.automain
def transcribe_classifier(single, player, class_dir, win_len, hop_len, gpu_num, seed, min_note_span):

    # Obtain the track list for the chosen data partition
    track_keys = clean_track_list(GuitarSetHandle, single, player, False)

    eval_tabs = GuitarSet(track_keys, win_len, hop_len, 'test', seed)

    loader = DataLoader(eval_tabs, 1, shuffle=False, num_workers=0, drop_last=False)

    device = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'

    classifier = torch.load(os.path.join(GEN_CLASS_DIR, f'{class_dir}/model.pt'))
    classifier.changeDevice(device)
    classifier.eval()

    cnt_wn = 9

    for clip in tqdm(loader):
        track_id = clip['id'][0]
        clip['cqt'], clip['tabs'] = clip['cqt'][0], clip['tabs'][0]
        pred, loss = classifier(clip)

        pred = pred.cpu().detach().numpy().T
        loss = loss.cpu().detach().numpy()

        midi_range = HIGHEST_NOTE - LOWEST_NOTE + 1

        num_frames = pred.shape[-1]
        tabs = np.zeros((NUM_STRINGS, midi_range, num_frames))

        for i in range(NUM_STRINGS):
            non_silent = pred[i] != NUM_FRETS + 1
            pitches = librosa.note_to_midi(TUNING[i]) + pred[i] - LOWEST_NOTE
            tabs[i, pitches[non_silent], non_silent] = 1

        frames = np.max(tabs, axis=0).T

        tab_notes = [extract_notes(tabs[i], win_len, hop_len, min_note_span) for i in range(NUM_STRINGS)]

        all_pitches, all_ints = [], []

        # Create the activation directory if it does not already exist
        reset_generated_dir(GEN_ESTIM_DIR, ['frames', 'notes', 'tabs'], False)

        tabs_dir = os.path.join(GEN_ESTIM_DIR, 'tabs', f'{track_id}')
        reset_generated_dir(tabs_dir, [], True)

        for i, s in enumerate(TUNING):
            tab_txt_path = os.path.join(tabs_dir, f'{s}.txt')
            pitches, ints = tab_notes[i]
            all_pitches += list(pitches)
            all_ints += list(ints)
            write_notes(tab_txt_path, pitches, ints)

        all_pitches, all_ints = np.array(all_pitches), np.array(all_ints)

        # Construct the paths for frame- and note-wise predictions
        frm_txt_path = os.path.join(GEN_ESTIM_DIR, 'frames', f'{track_id}.txt')
        nte_txt_path = os.path.join(GEN_ESTIM_DIR, 'notes', f'{track_id}.txt')

        # Save the predictions to file
        write_frames(frm_txt_path, win_len, hop_len, frames)
        write_notes(nte_txt_path, all_pitches, all_ints)
