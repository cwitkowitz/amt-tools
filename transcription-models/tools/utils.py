# My imports
from .constants import *

# Regular imports
from mir_eval.multipitch import resample_multipitch
from mir_eval.io import load_ragged_time_series
from mir_eval.io import load_valued_intervals
from copy import deepcopy

import numpy as np
import librosa
import random
import shutil
import torch
import jams
import math
import os

def seed_everything(seed):
    # WARNING: the number of workers in the training loader affects behavior:
    #          this is because each sample will inevitably end up being processed
    #          by a different worker if num_workers is changed, and each worker
    #          has its own random seed
    #          TODO - I will fix this in the future if possible

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def rms_norm(audio):
    rms = np.sqrt(np.mean(audio ** 2))

    assert rms != 0

    audio = audio / rms

    return audio

def load_audio(wav_path):
    audio, fs = librosa.load(wav_path, sr=None)
    assert fs == SAMPLE_RATE

    #audio = librosa.resample(audio, fs, SAMPLE_RATE)
    #audio = librosa.util.normalize(audio)# <- infinity norm

    audio = rms_norm(audio)

    return audio, fs

def framify_tfr(tfr, win_length, hop_length, pad=None):
    # TODO - avoid conversion in collate_fn instead?
    to_torch = False
    if 'torch' in str(tfr.dtype):
        to_torch = True
        tfr = tfr.cpu().detach().numpy()

    # TODO - parameterize axis or just assume -1?
    if pad is not None:
        pad_amts = [(0,)] * (len(tfr.shape) - 1) + [(pad,)]
        tfr = np.pad(tfr, tuple(pad_amts))

    #tfr = np.asfortranarray(tfr)
    # TODO - this is a cleaner solution but seems to be very unstable
    #stack = librosa.util.frame(tfr, win_length, hop_length).copy()

    dims = tfr.shape
    num_hops = (dims[-1] - 2 * pad) // hop_length
    hops = np.arange(0, num_hops, hop_length)
    new_dims = dims[:-1] + (win_length, num_hops)

    tfr = tfr.reshape(np.prod(dims[:-1]), dims[-1])
    tfr = [np.expand_dims(tfr[:, i : i + win_length], axis=-1) for i in hops]

    stack = np.concatenate(tfr, axis=-1)
    stack = np.reshape(stack, new_dims)

    if to_torch:
        stack = torch.from_numpy(stack)

    return stack

def track_to_batch(track):
    batch = deepcopy(track)

    batch['track'] = [batch['track']]
    batch['audio'] = torch.from_numpy(batch['audio']).unsqueeze(0)
    batch['tabs'] = torch.from_numpy(batch['tabs']).unsqueeze(0)
    batch['feats'] = torch.from_numpy(batch['feats']).unsqueeze(0)
    batch['notes'] = torch.from_numpy(batch['notes']).unsqueeze(0)

    return batch

def load_jams_guitar_tabs(jams_path, hop_length, num_frames):
    jam = jams.load(jams_path)

    # TODO - fails at 00_Jazz3-150-C_comp bc of an even division
    # duration = jam.file_metadata['duration']

    tabs = np.zeros((NUM_STRINGS, NUM_FRETS + 2, num_frames))

    frame_times = librosa.frames_to_time(range(num_frames), SAMPLE_RATE, hop_length)

    for s in range(NUM_STRINGS):
        string_notes = jam.annotations['note_midi'][s]
        frame_string_pitch = string_notes.to_samples(frame_times)

        silent = [pitch == [] for pitch in frame_string_pitch]

        tabs[s, -1, silent] = 1

        # TODO - is this for-loop absolutely necessary?
        for i in range(len(frame_string_pitch)):
            if silent[i]:
                frame_string_pitch[i] = [0]

        frame_string_pitch = np.array(frame_string_pitch).squeeze()

        open_string_midi_val = librosa.note_to_midi(TUNING[s])

        frets = frame_string_pitch[frame_string_pitch != 0] - open_string_midi_val
        frets = np.round(frets).astype('int')

        tabs[s, frets, frame_string_pitch != 0] = 1

    return tabs

def load_jams_guitar_contours(jams_path, hop_length):
    # TODO - should this even be a function? - are there even any other variants?
    pass

def load_jams_guitar_notes(jams_path):
    jam = jams.load(jams_path)

    i_ref = []
    p_ref = []

    for s in range(NUM_STRINGS):
        string_notes = jam.annotations['note_midi'][s]

        for note in string_notes:
            p_ref += [note.value]
            i_ref += [[note.time, note.time + note.duration]]

    # Obtain an array of time intervals for all note occurrences
    i_ref = np.array(i_ref)

    # Extract the ground-truth note pitch values into an array
    p_ref = librosa.midi_to_hz(np.array(p_ref))

    return i_ref, p_ref

def note_groups_to_arr(pitches, intervals):
    notes = None

    # TODO - list might not be the right default - it should be whatever the mir_eval note parsing function returns
    if len(pitches) > 0:
        # Batch-friendly note storage
        pitches = np.array([pitches]).T
        notes = np.concatenate((intervals, pitches), axis=-1)

    return notes

def arr_to_note_groups(note_arr):
    if note_arr is None:
        pitches, intervals = np.array([]), np.array([[], []]).T
    else:
        pitches, intervals = note_arr[:, -1], note_arr[:, :2]
    return pitches, intervals

def tabs_to_multi_pianoroll(tabs):
    num_frames = tabs.shape[-1]

    pianoroll = np.zeros((NUM_STRINGS, NOTE_RANGE, num_frames))

    for i in range(NUM_STRINGS):
        non_silent = tabs[i] != NUM_FRETS + 1

        pitches = librosa.note_to_midi(TUNING[i]) + tabs[i, non_silent] - LOWEST_NOTE

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
    pass

def get_pianoroll_offsets(pianoroll):
    pass

def pianoroll_to_pitchlist(pianoroll):
    active_pitches = []

    for i in range(pianoroll.shape[-1]): # For each frame
        # Determine the activations across this frame
        active_pitches += [librosa.midi_to_hz(np.where(pianoroll[:, i] != 0)[0] + LOWEST_NOTE)]

    return active_pitches

def write_and_print(file, text, verbose = True, end=''):
    try:
        # Try to write the text to the file (it is assumed to be open)
        file.write(text)
    finally:
        if verbose:
            # Print the text to console
            print(text, end=end)

def write_frames(path, h_len, pitches, places = 3):
    # Open a file at the path with writing permissions
    file = open(path, 'w')

    # Calculate the times corresponding to each frame prediction
    # TODO - window length is ambiguous - remove? - also check librosa func
    #times = (0.5 * w_len + np.arange(pitches.shape[0]) * h_len) / SAMPLE_RATE
    times = (np.arange(pitches.shape[0]) * h_len) / SAMPLE_RATE

    for i in range(times.size): # For each frame
        # Determine the activations across this frame
        active_pitches = librosa.midi_to_hz(np.where(pitches[i] != 0)[0] + LOWEST_NOTE)

        # Create a line of the form 'frame_time pitch1 pitch2 ...'
        line = str(round(times[i], places)) + ' ' + str(active_pitches)[1: -1]

        # Remove newline character
        line = line.replace('\n', '')

        # If we are not at the last line, add a newline character
        # TODO - can't I just change this to ==, remove \n instead, then I won't need the above line?
        if (i + 1) != times.size:
            line += '\n'

        # Write the line to the file
        file.write(line)

    # Close the file
    file.close()

def write_notes(path, pitches, intervals, places = 3):
    # Open a file at the path with writing permissions
    file = open(path, 'w')

    for i in range(len(pitches)): # For each note
        # Create a line of the form 'start_time end_time pitch'
        line = str(round(intervals[i][0], places)) + ' ' + \
               str(round(intervals[i][1], places)) + ' ' + \
               str(round(pitches[i], places))

        # Remove newline character
        line = line.replace('\n', '')

        # If we are not at the last line, add a newline character
        # TODO - can't I just change this to ==, remove \n instead, then I won't need the above line?
        if (i + 1) != len(pitches):
            line += '\n'

        # Write the line to the file
        file.write(line)

    # Close the file
    file.close()
