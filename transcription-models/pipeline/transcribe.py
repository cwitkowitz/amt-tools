# My imports
from tools.conversion import *
from tools.utils import *
from tools.io import *
from datasets.common import *

# Regular imports
import numpy as np
import torch
import os

# TODO - helper function to transcribe a singe piece of audio - turn it into track then batch (parameterize data_proc)

def extract_notes(frames, hop_len, sample_rate, min_note_span):
    # TODO - parameterize onsets optionally
    # TODO - clean this ish up
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
            if onset == offset and np.sum(onsets[:, max(0, onset - int(0.10 * sample_rate // hop_len)) : onset]) > 0:
                break
            offset += 1
            if offset == frames.shape[1]:
                break

        # Make sure the note duration exceeds a minimum frame length
        if offset >= onset + min_note_span:
            # Determine the absolute frequency
            freq = librosa.midi_to_hz(pitch + infer_lowest_note(frames))

            # Add the frequency to the list
            pitches.append(freq)

            # TODO - can probs utilize librosa here - it does same thing but with array
            # Determine the time where the onset and offset occur
            onset, offset = onset * hop_len / sample_rate, offset * hop_len / sample_rate

            # TODO - window length is ambiguous - remove? - also check librosa func
            # Add half of the window time for frame-centered predictions
            #bias = (0.5 * win_len / SAMPLE_RATE)
            #onset, offset = onset + bias, offset + bias

            # Add the interval to the list
            ints.append([onset, offset])

    # Convert the lists to numpy arrays
    pitches, intervals = np.array(pitches), np.array(ints)

    return pitches, intervals

def transcribe(classifier, track, hop_length, sample_rate, min_note_span, log_dir=None, tabs=False):
    # Just in case
    classifier.eval()

    with torch.no_grad():
        track = track_to_batch(track)
        preds, track_loss = classifier.run_on_batch(track)
        track_loss = torch.mean(track_loss).item()

        track_id = track['track'][0]

        # TODO - abstract the tab functionality - not all models are for guitar - for now a tabs boolean should be fine
        # TODO - remove unnecessary transpositions
        if tabs:
            tabs = preds.squeeze().cpu().detach().numpy().T

            pianoroll_by_string = tabs_to_multi_pianoroll(tabs)
            pianoroll = tabs_to_pianoroll(tabs)

            if log_dir is not None:
                os.makedirs(os.path.join(log_dir, 'tabs'), exist_ok=True)
                tabs_dir = os.path.join(log_dir, 'tabs', f'{track_id}')
                os.makedirs(os.path.join(tabs_dir), exist_ok=True)

            all_pitches, all_ints = [], []
            for i in range(NUM_STRINGS):
                pitches, ints = extract_notes(pianoroll_by_string[i], hop_length, sample_rate, min_note_span)
                all_pitches += list(pitches)
                all_ints += list(ints)

                if log_dir is not None:
                    tab_txt_path = os.path.join(tabs_dir, f'{i}.txt')
                    write_notes(tab_txt_path, pitches, ints)
        else:
            onsets, frames = preds
            pianoroll = frames.squeeze().cpu().detach().numpy()
            all_pitches, all_ints = extract_notes(pianoroll, hop_length, sample_rate, min_note_span)

        all_notes = note_groups_to_arr(all_pitches, all_ints)

        if log_dir is not None:
            os.makedirs(os.path.join(log_dir, 'frames'), exist_ok=True)
            os.makedirs(os.path.join(log_dir, 'notes'), exist_ok=True)

            # Construct the paths for frame- and note-wise predictions
            frm_txt_path = os.path.join(log_dir, 'frames', f'{track_id}.txt')
            nte_txt_path = os.path.join(log_dir, 'notes', f'{track_id}.txt')

            # Save the predictions to file
            write_frames(frm_txt_path, hop_length, sample_rate, pianoroll)
            write_notes(nte_txt_path, all_pitches, all_ints)

    predictions = {}

    predictions['track'] = track_id
    predictions['loss'] = track_loss
    predictions['pianoroll'] = pianoroll
    predictions['notes'] = all_notes

    return predictions
