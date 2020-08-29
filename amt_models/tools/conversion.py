# My imports
from tools.constants import *
from tools.utils import *

# Regular imports
from copy import deepcopy

import numpy as np
import librosa
import torch


def note_groups_to_arr(pitches, intervals):
    if len(pitches) > 0:
        # Batch-friendly note storage
        pitches = np.array([pitches]).T
        notes = np.concatenate((intervals, pitches), axis=-1)
    else:
        notes = np.array([[], [], []]).T

    return notes


def arr_to_note_groups(note_arr):
    # TODO - make sure this is consistent across usage - i.e. GuitarSet/MAPS/etc. - librosa.validate_intervals
    if note_arr is None:
        # TODO - this is a risky branch
        pitches, intervals = np.array([]), np.array([[], []]).T
    else:
        pitches, intervals = note_arr[:, -1], note_arr[:, :2]
    return pitches, intervals


def midi_groups_to_pianoroll(pitches, intervals, times, note_range):
    num_frames = times.size - 1
    num_notes = pitches.size
    pianoroll = np.zeros((note_range, num_frames))

    pitches = np.round(pitches - infer_lowest_note(pianoroll)).astype('uint')
    times = np.tile(np.expand_dims(times, axis=0), (num_notes, 1))

    onsets = np.argmin((times <= intervals[:, :1]), axis=1) - 1
    offsets = np.argmin((times < intervals[:, 1:]), axis=1) - 1

    # TODO - might be able to vectorize this
    for i in range(pitches.size):
        pianoroll[pitches[i], onsets[i] : offsets[i] + 1] = 1

    return pianoroll


# TODO - tabs to softmax function? - yes this was confusing
"""
tabs = batch['tabs'].transpose(1, 2)
tabs[tabs == -1] = NUM_FRETS + 1
tabs = torch.zeros(tabs_temp.shape + tuple([NUM_FRETS + 2]))
tabs = tabs.to(tabs_temp.device)

b, f, s = tabs_temp.size()
b, f, s = torch.meshgrid(torch.arange(b), torch.arange(f), torch.arange(s))
tabs[b, f, s, tabs_temp] = 1
tabs = tabs.view(-1, NUM_FRETS + 2).long()
"""

def tabs_to_multi_pianoroll(tabs):
    num_frames = tabs.shape[-1]

    pianoroll = np.zeros((NUM_STRINGS, GUITAR_RANGE, num_frames))

    # TODO - vectorize this
    for i in range(NUM_STRINGS):
        non_silent = tabs[i] != -1

        pitches = librosa.note_to_midi(TUNING[i]) + tabs[i, non_silent] - GUITAR_LOWEST

        pianoroll[i, pitches, non_silent] = 1

    return pianoroll


def tabs_to_pianoroll(tabs):
    pianoroll = tabs_to_multi_pianoroll(tabs)

    pianoroll = np.max(pianoroll, axis=0)

    return pianoroll


def get_multi_pianoroll_onsets(multi_pianoroll):
    multi_onsets = np.zeros(multi_pianoroll.shape)

    # TODO - verify onset in first frame
    # TODO - vectorize this
    for i in range(NUM_STRINGS):
        multi_onsets[i] = get_pianoroll_onsets(multi_pianoroll[i])

    return multi_onsets


def get_multi_pianoroll_offsets(pianoroll):
    # TODO - just use pianoroll function
    pass


# TODO - potentially fold notion of tabs into multi_pianoroll or vice versa
def multi_pianoroll_to_tabs(multi_pianoroll):
    num_frames = multi_pianoroll.shape[-1]

    # TODO - generalize num_strings to num_dofs?
    tabs = np.zeros((NUM_STRINGS, num_frames))

    no_note_row = np.ones((1, num_frames))

    for i in range(NUM_STRINGS):
        # TODO - generalize tuning to param
        start_idx = TUNING_MIDI[i, 0] - GUITAR_LOWEST
        pianoroll = multi_pianoroll[i, start_idx : start_idx + NUM_FRETS + 1]
        pianoroll = np.append(pianoroll, no_note_row, axis=0)
        tabs[i] = np.expand_dims(np.argmax(pianoroll, axis=0), axis=0)

    tabs[tabs == NUM_FRETS + 1] = -1
    return tabs


def get_pianoroll_onsets(pianoroll):
    first_frame = pianoroll[:, :1]
    adjacent_diff = pianoroll[:, 1:] - pianoroll[:, :-1]
    onsets = np.concatenate([first_frame, adjacent_diff], axis=1) == 1
    return onsets


def get_pianoroll_offsets(pianoroll):
    pass


def get_note_offsets(note_arr):
    pass


def pianoroll_to_pitchlist(pianoroll):
    active_pitches = []

    # TODO - vectorize?
    for i in range(pianoroll.shape[-1]): # For each frame
        # Determine the activations across this frame
        active_pitches += [librosa.midi_to_hz(np.where(pianoroll[:, i] != 0)[0] + infer_lowest_note(pianoroll))]

    return active_pitches


def to_single(activations):
    if valid_single(activations):
        pass
    elif valid_multi(activations):
        pass
    elif valid_tabs(activations):
        pass
    else:
        activations = NotImplementedError

    return activations


def to_multi(activations):
    if valid_single(activations):
        pass
    elif valid_multi(activations):
        pass
    elif valid_tabs(activations):
        pass
    else:
        activations = NotImplementedError

    return activations


def to_tabs(activations):
    if valid_single(activations):
        pass
    elif valid_multi(activations):
        pass
    elif valid_tabs(activations):
        pass
    else:
        activations = NotImplementedError

    return activations


def feats_to_batch(feats, times):
    # TODO - a function which accepts only feats (for deployment)
    pass


def track_to_batch(track):
    batch = deepcopy(track)

    if 'track' in batch:
        batch['track'] = [batch['track']]
    else:
        batch['track'] = ['no_name']

    keys = list(batch.keys())
    for key in keys:
        if isinstance(batch[key], np.ndarray):
            batch[key] = torch.from_numpy(batch[key]).unsqueeze(0)

    return batch


def track_to_dtype(track, dtype='float32'):
    track = deepcopy(track)

    keys = list(track.keys())
    for key in keys:
        if isinstance(track[key], np.ndarray):
            track[key] = track[key].astype(dtype)

    return track


def track_to_device(track, device):
    keys = list(track.keys())

    for key in keys:
        if isinstance(track[key], torch.Tensor):
            track[key] = track[key].to(device)

    return track


def track_to_cpu(track):
    keys = list(track.keys())

    for key in keys:
        if isinstance(track[key], torch.Tensor):
            track[key] = track[key].squeeze().cpu().detach().numpy()

    return track
