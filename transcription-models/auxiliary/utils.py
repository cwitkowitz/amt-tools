# My imports
from .constants import *

# Regular imports
from mir_eval.multipitch import resample_multipitch
from mir_eval.io import load_ragged_time_series
from mir_eval.io import load_valued_intervals

import numpy as np
import librosa
import random
import shutil
import torch
import jams
import math
import os

def seed_everything(seed):
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
    #audio = librosa.util.normalize(audio) <- infinity norm

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

def extract_notes(frames, hop_len, min_note_span):
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

            # TODO - can probs utilize librosa here - it does same thing but with array
            # Determine the time where the onset and offset occur
            onset, offset = onset * hop_len / SAMPLE_RATE, offset * hop_len / SAMPLE_RATE

            # TODO - window length is ambiguous - remove? - also check librosa func
            # Add half of the window time for frame-centered predictions
            #bias = (0.5 * win_len / SAMPLE_RATE)
            #onset, offset = onset + bias, offset + bias

            # Add the interval to the list
            ints.append([onset, offset])

    # Convert the lists to numpy arrays
    pitches, intervals = np.array(pitches), np.array(ints)

    return pitches, intervals

########################################
# TODO - Re-verify everything underneath
########################################

def sample_pitches(new_times, orig_times, orig_pitches):
    new_pitches = []
    for i in range(len(new_times)):
        closest_time_ind = np.argmin(abs(orig_times - new_times[i]))
        new_pitches += [orig_pitches[closest_time_ind]]

    return new_times, new_pitches

def pitches_to_arr(times, freqs):
    idcs = [freqs[i].size != 0 for i in range(len(freqs))]
    times, freqs = times[idcs], np.array(freqs)[idcs]

    t_nz, f_nz = [], []

    [t_nz.extend([times[i]] * freqs[i].size) for i in range(freqs.shape[0])]
    [f_nz.extend(list(freqs[i])) for i in range(freqs.shape[0])]

    times, freqs = np.array(t_nz), np.array(f_nz)

    return times, freqs

def jams_to_pitch_arr(jamsData, places = 3):
    # Get the pitch annotations for each string from the JAMS file
    sources = jamsData.search(namespace = 'pitch_contour')

    # Construct a dictionary to hold all pitches across time
    pitch_dict = {}

    for string in sources:
        # Construct a dictionary to hold the string pitches across time
        source_dict = {}

        # Extract the pitch annotations from the JAMS data
        true_annos = jams.eval.coerce_annotation(string, namespace = 'pitch_contour')

        # Obtain the intervals and pitch values of the annotations
        ints_true, pitches_true = true_annos.to_interval_values()

        # Obtain the starting times of all the pitches
        times_true = ints_true[:, 0]

        # Obtain all active pitch values
        pitches_true = np.asarray([p['frequency'] for p in pitches_true])

        # Add all of the [time, pitch(es)] pairs to the string dictionary
        source_dict = add_pitches_to_dict(source_dict, times_true, pitches_true)

        # Convert the string dictionary to an array
        times_true, pitches_true = dict_to_pitches(source_dict)

        # Round the times to a specified decimal and throw away repeats to get targets
        times_trgt = np.unique(np.round(times_true, places))

        # Resample the pitches to aligned with the target times
        pitches_aligned = resample_multipitch(times_true, pitches_true, times_trgt)

        # Add the resampled [time, pitch(es)] pairs to the pitch dictionary
        pitch_dict = add_pitches_to_dict(pitch_dict, times_trgt, pitches_aligned, places)

    # Add null entries for missing times
    pitch_dict = fill_in_dict_with_empties(pitch_dict, places)

    # Get a final array of times and pitches
    times, pitches = dict_to_pitches(pitch_dict)

    return times, pitches

def add_pitches_to_dict(pitch_dict, times, pitches, places = 0):
    # Loop through all of the given times
    for i in range(times.size):
        if places == 0:
            # Do not round
            t = times[i]
        else:
            # Round to the specified decimal place
            t = round(times[i], places)

        if t in pitch_dict:
            # Append the new pitch to the existing entry
            pitch_dict[t] = np.append(pitch_dict[t], pitches[i])
        else:
            # Create a new entry for the pitch
            pitch_dict[t] = np.append(np.array([]), pitches[i])

    return pitch_dict

def dict_to_pitches(pitch_dict):
    # Obtain the times (keys) and active pitches (values) of the dictionary
    times, pitches = list(pitch_dict.keys()), list(pitch_dict.values())

    # Sort the active pitch lists by time
    pitches = [p for _, p in sorted(zip(times, pitches))]

    # Sort the times and convert them to an array
    times = np.array(sorted(times))

    # Remove any zero pitch entries within the pitch lists
    for i in range(len(pitches)):
        pitches[i] = pitches[i][pitches[i] != 0]

    return times, pitches

def fill_in_dict_with_empties(pitch_dict, places = 2, t_start = 0., t_end = -1):
    # Time resolution for entries
    t_step = math.pow(10, -places)

    if t_end == -1:
        # Fill up with times until the integer ceiling of the highest time recorded
        t_end = int(math.ceil(max(pitch_dict.keys())))

    # Loop through the chosen starting and ending time
    for t in np.arange(t_start, t_end, t_step):
        # Remove trailing zeros
        t = round(t, places)

        # Associate a null array with the time in the dictionary if there is no entry
        if t not in pitch_dict:
            pitch_dict[t] = np.array([])

    return pitch_dict

def write_and_print(file, text, verbose = True):
    try:
        # Try to write the text to the file (it is assumed to be open)
        file.write(text)
    finally:
        if verbose:
            # Print the text to console
            print(text, end = '')

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

    for i in range(pitches.size): # For each note
        # Create a line of the form 'start_time end_time pitch'
        line = str(round(intervals[i, 0], places)) + ' ' + \
               str(round(intervals[i, 1], places)) + ' ' + \
               str(round(pitches[i], places))

        # Remove newline character
        line = line.replace('\n', '')

        # If we are not at the last line, add a newline character
        # TODO - can't I just change this to ==, remove \n instead, then I won't need the above line?
        if (i + 1) != pitches.size:
            line += '\n'

        # Write the line to the file
        file.write(line)

    # Close the file
    file.close()

def get_tabs(id):
    track = GuitarSetHandle[id]  # Track data handle

    tabs_dir = os.path.join(GEN_ESTIM_DIR, 'tabs', f'{id}')

    tabs_est = []

    for str_name in TUNING:
        str_txt_path = os.path.join(tabs_dir, f'{str_name}.txt')
        ints, pitches = load_valued_intervals(str_txt_path)
        tabs_est += [[pitches, ints]]

    notes = track.notes

    tabs_ref = []

    for str_name in notes.keys():
        pitches = librosa.midi_to_hz(np.array(notes[str_name].notes))

        t_st = notes[str_name].start_times
        t_fn = notes[str_name].end_times

        ints = np.array([[t_st[i], t_fn[i]] for i in range(pitches.size)])

        tabs_ref += [[pitches, ints]]

    return tabs_est, tabs_ref
