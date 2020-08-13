# My imports
from tools.conversion import *
from tools.io import *

from datasets.common import *

# Regular imports
import numpy as np
import torch
import os


def get_num_nzs(arr):
    return len(arr.nonzero()[0])


def filter_short_notes(pitches, intervals, t_trh=0.250):
    note_arr = note_groups_to_arr(pitches, intervals)
    durations = note_arr[:, 1] - note_arr[:, 0]
    note_arr = note_arr[durations > t_trh]
    pitches, intervals = arr_to_note_groups(note_arr)
    return pitches, intervals


def inhibit_activations(activations, times, t_window=0.050):
    num_nz_prev = 0

    while num_nz_prev != get_num_nzs(activations):
        num_nz_prev = get_num_nzs(activations)
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


def predict_notes(frames, times, onsets=None, hard_inhibition=True, filter_length=True):
    if onsets is None:
        onsets = get_pianoroll_onsets(frames)

    if hard_inhibition:
        onsets = inhibit_activations(onsets, times, 0.10)

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
        pitches.append(librosa.midi_to_hz(pitch + infer_lowest_note(frames)))

        # Add the interval to the list
        onset, offset = times[onset], times[offset]
        ints.append([onset, offset])

    # Convert the lists to numpy arrays
    pitches, intervals = np.array(pitches), np.array(ints)

    # Remove all notes which only last one frame
    pitches, intervals = filter_short_notes(pitches, intervals, 0.0)

    if filter_length:
        pitches, intervals = filter_short_notes(pitches, intervals, 0.250)

    return pitches, intervals

def transcribe(model, track, log_dir=None):
    # Just in case
    model.eval()

    with torch.no_grad():
        batch = track_to_batch(track)
        preds = model.run_on_batch(batch)
        track_loss = torch.mean(preds['loss']).item()

        track_id = track['track']

        times = track['times']

        pianoroll = None
        if 'pianoroll' in preds.keys():
            pianoroll = preds['pianoroll'].squeeze().cpu().detach().numpy()

        onsets = None
        if 'onsets' in preds.keys():
            onsets = preds['onsets'].squeeze().cpu().detach().numpy()

        if 'tabs' in preds.keys():
            tabs = preds['tabs'].squeeze().cpu().detach().numpy()

            pianoroll_by_string = tabs_to_multi_pianoroll(tabs)
            pianoroll = tabs_to_pianoroll(tabs)

            if log_dir is not None:
                os.makedirs(os.path.join(log_dir, 'tabs'), exist_ok=True)
                tabs_dir = os.path.join(log_dir, 'tabs', f'{track_id}')
                os.makedirs(os.path.join(tabs_dir), exist_ok=True)

            all_pitches, all_ints = [], []
            for i in range(NUM_STRINGS):
                pitches, ints = predict_notes(pianoroll_by_string[i], times)
                all_pitches += list(pitches)
                all_ints += list(ints)

                if log_dir is not None:
                    tab_txt_path = os.path.join(tabs_dir, f'{i}.txt')
                    write_notes(tab_txt_path, pitches, ints)
        else:
            all_pitches, all_ints = predict_notes(pianoroll, times, onsets)

        all_notes = note_groups_to_arr(all_pitches, all_ints)

        if log_dir is not None:
            os.makedirs(os.path.join(log_dir, 'frames'), exist_ok=True)
            os.makedirs(os.path.join(log_dir, 'notes'), exist_ok=True)

            # Construct the paths for frame- and note-wise predictions
            frm_txt_path = os.path.join(log_dir, 'frames', f'{track_id}.txt')
            nte_txt_path = os.path.join(log_dir, 'notes', f'{track_id}.txt')

            # Save the predictions to file
            write_frames(frm_txt_path, pianoroll, times[:-1])
            write_notes(nte_txt_path, all_pitches, all_ints)


    preds['track'] = track_id
    preds['loss'] = track_loss
    preds['times'] = times
    preds['notes'] = all_notes

    return preds
