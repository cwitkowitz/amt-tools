# My imports
from tools.constants import *
from tools.utils import *

# Regular imports
from copy import deepcopy

import numpy as np
import librosa


def note_groups_to_arr(pitches, intervals):
    notes = None

    # TODO - list might not be the right default - it should be whatever the mir_eval note parsing function returns
    if len(pitches) > 0:
        # Batch-friendly note storage
        pitches = np.array([pitches]).T
        notes = np.concatenate((intervals, pitches), axis=-1)

    return notes


def arr_to_note_groups(note_arr):
    # TODO - make sure this is consistent across usage - i.e. GuitarSet/MAPS/etc. - librosa.validate_intervals
    if note_arr is None:
        # TODO - this is a pad branch
        pitches, intervals = np.array([]), np.array([[], []]).T
    else:
        pitches, intervals = note_arr[:, -1], note_arr[:, :2]
    return pitches, intervals


# TODO - are hl and sr actually required? - no times as input
def note_groups_to_pianoroll(pitches, intervals, hop_length, sample_rate, note_range, num_frames):
    # Expects MIDI format
    pianoroll = np.zeros((note_range, num_frames))
    intervals = np.round(intervals * sample_rate / hop_length).astype('uint')
    intervals[:, 1] = intervals[:, 1] + 1

    pitches = np.round(pitches - infer_lowest_note(pianoroll)).astype('uint')

    # TODO - might be able to vectorize this
    for i in range(pitches.size):
        pianoroll[pitches[i], intervals[i, 0] : intervals[i, 1]] = 1

    return pianoroll


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


def get_multi_pianoroll_onsets(pianoroll):
    # TODO - just use pianoroll function
    pass


def get_multi_pianoroll_offsets(pianoroll):
    # TODO - just use pianoroll function
    pass


def get_pianoroll_onsets(pianoroll, dtype='uint'):
    first_frame = pianoroll[:, :1]
    adjacent_diff = pianoroll[:, 1:] - pianoroll[:, :-1]
    onsets = np.concatenate([first_frame, adjacent_diff], axis=1) == 1

    # TODO - uint64 breaks collate_fn - maybe put this is one of the batching functions as well
    onsets = onsets.astype(dtype)
    return onsets


def get_pianoroll_offsets(pianoroll):
    pass


def get_note_offsets(note_arr):
    pass


def pianoroll_to_pitchlist(pianoroll):
    active_pitches = []

    for i in range(pianoroll.shape[-1]): # For each frame
        # Determine the activations across this frame
        active_pitches += [librosa.midi_to_hz(np.where(pianoroll[:, i] != 0)[0] + infer_lowest_note(pianoroll))]

    return active_pitches


# TODO - a function which accepts only feats (for deployment) would be nice
def track_to_batch(track):
    batch = deepcopy(track)

    # TODO - if no name provided, give it a dummy name
    batch['track'] = [batch['track']]

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
