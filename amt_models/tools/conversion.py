# My imports
#from tools.instrument import *
from tools.constants import *
from tools.utils import *

# Regular imports
from copy import deepcopy

import numpy as np
import librosa
import torch


# TODO - major cleanup needed for all of these functions

def note_groups_to_arr(pitches, intervals):
    # TODO - validate prior to conversion?
    if len(pitches) > 0:
        # Batch-friendly note storage
        pitches = np.array([pitches]).T
        notes = np.concatenate((intervals, pitches), axis=-1)
    else:
        notes = np.array([[], [], []]).T

    return notes


def arr_to_note_groups(note_arr):
    if note_arr is None:
        # TODO - this is a risky branch - can't remember why
        pitches, intervals = np.array([]), np.array([[], []]).T
    else:
        pitches, intervals = note_arr[:, -1], note_arr[:, :2]
    assert valid_notes(pitches, intervals)
    return pitches, intervals


def midi_groups_to_pianoroll(pitches, intervals, times, note_range):
    num_frames = times.size - 1
    num_notes = pitches.size
    pianoroll = np.zeros((note_range.size, num_frames))

    pitches = np.round(pitches - note_range[0]).astype('uint')
    times = np.tile(np.expand_dims(times, axis=0), (num_notes, 1))

    onsets = np.argmin((times <= intervals[:, :1]), axis=1) - 1
    offsets = np.argmin((times < intervals[:, 1:]), axis=1) - 1

    # TODO - might be able to vectorize this
    for i in range(pitches.size):
        pianoroll[pitches[i], onsets[i] : offsets[i] + 1] = 1

    return pianoroll


def tabs_to_multi_pianoroll(tabs, profile):
    # TODO - for now make sure it's guitar - can generalize in future
    #assert isinstance(profile, GuitarProfile)

    tabs = tabs.copy().astype('int')

    shape = list(tabs.shape) + [profile.get_range_len()]
    #shape = list(tabs.shape)
    #shape = shape[:-1] + [profile.get_range_len()] + shape[-1:]
    pianoroll = np.zeros(shape)

    midi_tuning = profile.get_midi_tuning()
    multi_start = midi_tuning - profile.low

    if shape[0] != profile.num_strings:
        # Add a batch dimension
        multi_start = np.expand_dims(multi_start, axis=0)
        multi_start = np.repeat(multi_start, shape[0], axis=0)

    non_silent = tabs != -1

    pitches = tabs + multi_start
    pitches = pitches[non_silent]

    idcs = tuple(list(non_silent.nonzero()) + [pitches])

    pianoroll[idcs] = 1
    pianoroll = np.swapaxes(pianoroll, -1, -2)

    return pianoroll


def tabs_to_pianoroll(tabs):
    pianoroll = tabs_to_multi_pianoroll(tabs)

    pianoroll = np.max(pianoroll, axis=0)

    return pianoroll


def get_onsets(pitch, profile):
    to_torch = 'torch' in str(pitch.dtype)
    if to_torch:
        device = pitch.device
        pitch = pitch.cpu().detach().numpy()

    tabs = False
    onsets = None

    # TODO - how to deal with batch dimension properly?
    if valid_tabs(pitch[0], profile):
        pitch = tabs_to_multi_pianoroll(pitch, profile)
        tabs = True

    if valid_multi(pitch[0], profile) or valid_single(pitch[0], profile):
        onsets = get_pianoroll_onsets(pitch)

    if tabs:
        onsets = multi_pianoroll_to_tabs(onsets, profile)

    if to_torch:
        onsets = torch.from_numpy(onsets)
        onsets = onsets.to(device)

    return onsets


def multi_pianoroll_to_tabs(multi_pianoroll, profile):
    # TODO - for now make sure it's guitar - can generalize (num_dofs) in future
    #assert isinstance(profile, GuitarProfile)

    shape = list(multi_pianoroll.shape)
    silent_idx = shape.pop(-2)

    no_note_row = np.ones(shape)
    no_note_row = np.expand_dims(no_note_row, axis=-2)
    multi_pianoroll = np.append(multi_pianoroll, no_note_row, axis=-2)

    tabs = np.argmax(multi_pianoroll, axis=-2)

    silent = tabs == silent_idx

    midi_tuning = profile.get_midi_tuning()
    multi_start = midi_tuning - profile.low

    if shape[0] != profile.num_strings:
        # Add a batch dimension
        multi_start = np.expand_dims(multi_start, axis=0)
        multi_start = np.repeat(multi_start, shape[0], axis=0)

    tabs = tabs - multi_start

    tabs[silent] = -1
    return tabs


def get_pianoroll_onsets(pianoroll):
    first_frame = pianoroll[..., :1]
    adjacent_diff = pianoroll[..., 1:] - pianoroll[..., :-1]
    onsets = np.concatenate([first_frame, adjacent_diff], axis=-1) == 1
    return onsets


def get_pianoroll_offsets(pianoroll):
    pass


def get_note_offsets(note_arr):
    pass


def pianoroll_to_pitchlist(pianoroll, lowest):
    active_pitches = []

    # TODO - vectorize?
    for i in range(pianoroll.shape[-1]): # For each frame
        # Determine the activations across this frame
        active_pitches += [librosa.midi_to_hz(np.where(pianoroll[:, i] != 0)[0] + lowest)]

    return active_pitches


def to_single(activations, profile):
    if valid_tabs(activations, profile):
        activations = tabs_to_multi_pianoroll(activations, profile)

    if valid_multi(activations, profile):
        # TODO - multi_to_single and single_to_multi funcs
        activations = np.max(activations, axis=0)

    assert valid_single(activations, profile)

    return activations


def to_multi(activations, profile):
    if valid_single(activations, profile):
        activations = np.expand_dims(activations, axis=0)

    if valid_tabs(activations, profile):
        activations = tabs_to_multi_pianoroll(activations, profile)

    assert valid_multi(activations, profile)

    return activations


def to_tabs(activations, profile):
    if valid_single(activations, profile):
        activations = np.expand_dims(activations, axis=0)

    if valid_multi(activations, profile):
        activations = multi_pianoroll_to_tabs(activations, profile)

    assert valid_tabs(activations, profile)

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
