# My imports
from tools.constants import *
from tools.utils import *

# Regular imports
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
    # TODO - make sure this is consistent across usage - i.e. GuitarSet/MAPS/etc.
    if note_arr is None:
        pitches, intervals = np.array([]), np.array([[], []]).T
    else:
        pitches, intervals = note_arr[:, -1], note_arr[:, :2]
    return pitches, intervals


def note_groups_to_pianoroll(pitches, intervals, hop_length, sample_rate, note_range, num_frames):
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

    for i in range(NUM_STRINGS):
        non_silent = tabs[i] != NUM_FRETS + 1

        pitches = librosa.note_to_midi(TUNING[i]) + tabs[i, non_silent] - GUITAR_LOWEST

        pianoroll[i, pitches, non_silent] = 1

    return pianoroll


def tabs_to_pianoroll(tabs):
    pianoroll = tabs_to_multi_pianoroll(tabs)

    pianoroll = np.max(pianoroll, axis=0)

    return pianoroll


def get_tab_onsets(tabs):
    # TODO - just use pianoroll function
    pass


def get_tab_offsets(tabs):
    # TODO - just use pianoroll function
    pass


def get_pianoroll_onsets(pianoroll):
    onsets = np.concatenate([pianoroll[:, :1], pianoroll[:, 1:] - pianoroll[:, :-1]], axis=1) == 1
    # TODO - uint64 breaks collate_fn
    onsets = onsets.astype('float64')
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
