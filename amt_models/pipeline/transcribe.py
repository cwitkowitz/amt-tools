# My imports
from tools.conversion import *
from tools.utils import *
from tools.io import *

from datasets.common import *

# Regular imports
import numpy as np
import torch
import os


def filter_short_notes(pitches, intervals, t_trh=0.250):
    note_arr = note_groups_to_arr(pitches, intervals)
    durations = note_arr[:, 1] - note_arr[:, 0]
    note_arr = note_arr[durations > t_trh]
    pitches, intervals = arr_to_note_groups(note_arr)
    return pitches, intervals


def inhibit_activations(activations, times, t_window=0.050):
    num_nz_prev = 0

    while num_nz_prev != len(activations.nonzero()[0]):
        num_nz_prev = len(activations.nonzero()[0])
        # TODO - could remove nonzeros which have already been processed
        nzs = activations.nonzero()
        for pitch, inhib_start in zip(nzs[0], nzs[1]):
            in_bounds = np.where(times > times[inhib_start] + t_window)[0]
            if len(in_bounds) > 0:
                inhib_end = in_bounds[0]
                # TODO - watch for out of bounds error
                if sum(activations[pitch, inhib_start + 1 : inhib_end]) > 0:
                    activations[pitch, inhib_start + 1: inhib_end] = 0
                    break

    return activations


# TODO - remember to turn off these params when attempting SOTA recreation
def predict_notes(frames, times, lowest=0, onsets=None, hard_inhibition=False, filter_length=False):
    if onsets is None:
        onsets = get_pianoroll_onsets(frames)
    #else:
    #    assert valid_single(onsets)

    if hard_inhibition:
        onsets = inhibit_activations(onsets, times, 0.05)

    # Create empty lists for note pitches and their time intervals
    pitches, ints = [], []

    # Find the nonzero indices
    nzs = onsets.nonzero()
    for pitch, frame in zip(nzs[0], nzs[1]):
        # Mark onset and start offset counter
        onset, offset = frame, frame

        # Increment the offset counter until the pitch activation
        # turns negative or until the last frame is reached
        while frames[pitch, offset] and not (onsets[pitch, offset] and onset != offset):
            offset += 1
            if offset == frames.shape[1]:
                break

        # Add the frequency to the list
        pitches.append(librosa.midi_to_hz(pitch + lowest))

        # Add the interval to the list
        onset, offset = times[onset], times[offset]
        ints.append([onset, offset])

    # Convert the lists to numpy arrays
    pitches, intervals = np.array(pitches), np.array(ints)

    # Sort by onset just for the purpose of being neat - won't affect results
    note_arr = note_groups_to_arr(pitches, intervals)
    note_arr = note_arr[np.argsort(note_arr[:, 0])]
    pitches, intervals = arr_to_note_groups(note_arr)

    # Remove all notes which only last one frame
    one_frame_time = times[1] - times[0]
    pitches, intervals = filter_short_notes(pitches, intervals, one_frame_time)

    if filter_length:
        pitches, intervals = filter_short_notes(pitches, intervals, 0.050)

    return pitches, intervals


def predict_multi(pitch_multi, times, lowest=0, onsets_multi=None):
    multi_num = pitch_multi.shape[0]

    if onsets_multi is None:
        onsets_multi = [None] * multi_num
    #else:
    #    assert valid_multi(onsets_multi)

    notes_multi = []
    for i in range(multi_num):
        pitches, intervals = predict_notes(pitch_multi[i], times, lowest, onsets_multi[i])

        notes_multi += [[pitches, intervals]]

    return notes_multi


def transcribe(model, track, profile, log_dir=None):
    # Just in case
    model.eval()

    with torch.no_grad():
        batch = track_to_batch(track)
        preds = model.run_on_batch(batch)
        preds = track_to_cpu(preds)

        track_id = track[TR_ID]
        preds[TR_ID] = track_id

        preds['loss'] = preds['loss'].item()

        times = track[TIMES]
        preds[TIMES] = times

        pitch = None
        if PITCH in preds.keys():
            pitch = preds[PITCH]
            preds.pop(PITCH)

        onsets = None
        if ONSET in preds.keys():
            onsets = preds[ONSET]
            preds.pop(ONSET)

        if pitch is not None and valid_activations(pitch, profile):
            pitch = to_multi(pitch, profile)

            if onsets is not None:
                onsets = to_multi(onsets, profile)

            if valid_multi(pitch, profile):
                preds[MULT_PITCH] = pitch

                if log_dir is not None:
                    pitch_dir = os.path.join(log_dir, f'{track_id}', MULT_PITCH)
                    write_pitch_multi(pitch_dir, pitch, times[:-1], profile.low, profile.tuning)

                notes_multi = predict_multi(pitch, times, profile.low, onsets)
                preds[MULT_NOTES] = notes_multi

                if log_dir is not None:
                    notes_dir = os.path.join(log_dir, f'{track_id}', MULT_NOTES)
                    write_notes_multi(notes_dir, notes_multi, profile.tuning)

                pitch = to_single(pitch, profile)
                if onsets is not None:
                    onsets = to_single(onsets, profile)

            if valid_single(pitch, profile):
                preds[SOLO_PITCH] = pitch

                if log_dir is not None:
                    pitch_path = os.path.join(log_dir, f'{track_id}', f'{SOLO_PITCH}.txt')
                    write_pitch(pitch_path, pitch, times[:-1], profile.low)

                note_pitches, note_intervals = predict_notes(pitch, times, profile.low, onsets)
                preds[SOLO_NOTES] = (note_pitches, note_intervals)

                if log_dir is not None:
                    notes_path = os.path.join(log_dir, f'{track_id}', f'{SOLO_NOTES}.txt')
                    write_notes(notes_path, note_pitches, note_intervals)

        # TODO - option to redo pianoroll from note predictions

    return preds
