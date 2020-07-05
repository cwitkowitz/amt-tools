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

    tfr = np.asfortranarray(tfr)
    # TODO - this seems to be very unstable when called inside forward()
    stack = librosa.util.frame(tfr, win_length, hop_length).copy()

    if to_torch:
        stack = torch.from_numpy(stack)

    return stack

def load_jams_guitar_notes(jams_path, hop_length):
    jam = jams.load(jams_path)

    duration = jam.file_metadata['duration']

    num_frames = 1 + int((SAMPLE_RATE * duration - 1) // hop_length)

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

def tabs_to_pianoroll(tabs):
    num_frames = tabs.shape[-1]

    tabs = np.argmax(tabs, axis=1)

    pianoroll = np.zeros((NUM_STRINGS, NOTE_RANGE, num_frames))

    for i in range(NUM_STRINGS):
        non_silent = tabs[i] != NUM_FRETS + 1

        pitches = librosa.note_to_midi(TUNING[i]) + tabs[i, non_silent] - LOWEST_NOTE

        pianoroll[i, pitches, non_silent] = 1

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

def load_jams_guitar_contours(jams_path, hop_length):
    # TODO - should this even be a function? - are there even any other variants?
    pass

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

def write_frames(path, w_len, h_len, pitches, places = 3):
    # Open a file at the path with writing permissions
    file = open(path, 'w')

    # Calculate the times corresponding to each frame prediction
    # TODO - librosa here
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

def get_frames_contours(id, hop_len):
    track = GuitarSetHandle[id]  # Track data handle

    # Get the path to the frame-wise pitch estimations
    frm_txt_path = os.path.join(GEN_ESTIM_DIR, 'frames', f'{id}.txt')

    # Load the frame-wise pitch estimations
    t_est, f_est = load_ragged_time_series(frm_txt_path)

    #frames = track.pitch_contours

    # Load the ground-truth predictions and convert from JAMS to arrays
    # TODO - possibly leverage track data instead of my convoluted JAMS reading approach
    #t_ref = np.concatenate([frames[key].times for key in frames.keys()])
    #f_ref = np.concatenate([frames[key].frequencies for key in frames.keys()])
    #t_ref, f_ref = jams_to_pitch_arr(jams.load(track.jams_path), 2)
    #t_ref, f_ref = sample_pitches(t_est, t_ref, f_ref)

    num_frames = len(t_est)
    tabs = np.zeros((NUM_STRINGS, NUM_FRETS + 2, num_frames))

    jam = jams.load(track.jams_path)
    frame_indices = range(num_frames)
    t_ref = librosa.frames_to_time(frame_indices, SAMPLE_RATE, hop_len)

    for s in range(NUM_STRINGS):
        anno = jam.annotations['note_midi'][s]
        pitch = anno.to_samples(t_ref)
        silent = [pitch[i] == [] for i in range(len(pitch))]
        tabs[s, -1, silent] = 1
        for i in range(len(pitch)):
            if silent[i]:
                pitch[i] = [0]
        pitch = np.array(pitch).squeeze()
        midi_pitches = (np.round(pitch[pitch != 0] - librosa.note_to_midi(TUNING[s]))).astype('uint32')
        tabs[s, midi_pitches, pitch != 0] = 1

    tabs = np.argmax(tabs, axis=1).astype('uint32')
    midi_range = HIGHEST_NOTE - LOWEST_NOTE + 1
    frames = np.zeros((NUM_STRINGS, midi_range, num_frames))

    for i in range(NUM_STRINGS):
        non_silent = tabs[i] != NUM_FRETS + 1
        pitches = librosa.note_to_midi(TUNING[i]) + tabs[i] - LOWEST_NOTE
        frames[i, pitches[non_silent], non_silent] = 1

    frames = np.max(frames, axis=0).T

    t_ref = t_est
    f_ref = []

    for i in range(t_ref.size): # For each frame
        # Determine the activations across this frame
        f_ref += [librosa.midi_to_hz(np.where(frames[i] != 0)[0] + LOWEST_NOTE)]

    return t_est, f_est, t_ref, f_ref

# TODO - split into two and re-use in train/transcribe
def get_frames_notes(id, hop_len):
    track = GuitarSetHandle[id]  # Track data handle

    # Get the path to the frame-wise pitch estimations
    frm_txt_path = os.path.join(GEN_ESTIM_DIR, 'frames', f'{id}.txt')

    # Load the frame-wise pitch estimations
    t_est, f_est = load_ragged_time_series(frm_txt_path)

    notes = track.notes

    num_frames = len(t_est)
    midi_range = HIGHEST_NOTE - LOWEST_NOTE + 1

    tabs = np.zeros((NUM_STRINGS, NUM_FRETS + 2, num_frames))

    for i, s_key in enumerate(notes.keys()):
        s_data = notes[s_key]
        onset = np.round((s_data.start_times * SAMPLE_RATE) // hop_len).astype('uint32')
        offset = np.round((s_data.end_times * SAMPLE_RATE) // hop_len).astype('uint32')

        fret = np.round(np.array(s_data.notes) - librosa.note_to_midi(TUNING[i])).astype('uint32')

        for n in range(len(fret)):
            tabs[i, fret[n], onset[n]:offset[n]] = 1

        tabs[i, -1, np.sum(tabs[i], axis=0) == 0] = 1

    tabs = np.argmax(tabs, axis=1).astype('uint32')

    frames = np.zeros((NUM_STRINGS, midi_range, num_frames))

    for i in range(NUM_STRINGS):
        non_silent = tabs[i] != NUM_FRETS + 1
        pitches = librosa.note_to_midi(TUNING[i]) + tabs[i] - LOWEST_NOTE
        frames[i, pitches[non_silent], non_silent] = 1

    frames = np.max(frames, axis=0).T

    t_ref = t_est
    f_ref = []

    for i in range(t_ref.size): # For each frame
        # Determine the activations across this frame
        f_ref += [librosa.midi_to_hz(np.where(frames[i] != 0)[0] + LOWEST_NOTE)]

    return t_est, f_est, t_ref, f_ref

def get_notes(id):
    track = GuitarSetHandle[id]  # Track data handle

    # Get the path to the note-wise estimations
    nte_txt_path = os.path.join(GEN_ESTIM_DIR, 'notes', f'{id}.txt')

    # Load the note-wise estimations
    i_est, p_est = load_valued_intervals(nte_txt_path)

    notes = track.notes  # Dictionary of notes for this track

    # Extract the ground-truth note start times into an array
    i_ref_st = np.concatenate([notes[key].start_times for key in notes.keys()])

    # Extract the ground-truth note end times into an array
    i_ref_fn = np.concatenate([notes[key].end_times for key in notes.keys()])

    # Obtain an array of time intervals for all note occurrences
    i_ref = np.array([[i_ref_st[i], i_ref_fn[i]] for i in range(len(i_ref_st))])

    # Extract the ground-truth note pitch values into an array
    p_ref = librosa.midi_to_hz(np.concatenate([notes[key].notes for key in notes.keys()]))

    return i_est, p_est, i_ref, p_ref

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
