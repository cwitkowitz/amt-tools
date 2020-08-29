# My imports
from tools.constants import *
from tools.utils import *

# Regular imports
import numpy as np
import librosa
import jams


def load_audio(wav_path, sample_rate=None):
    audio, fs = librosa.load(wav_path, sr=sample_rate)

    #audio = librosa.util.normalize(audio)# <- infinity norm
    audio = rms_norm(audio)

    return audio, fs


def load_jams_guitar_tabs(jams_path, times):
    jam = jams.load(jams_path)

    # TODO - duration fails at 00_Jazz3-150-C_comp bc of an even division
    # duration = jam.file_metadata['duration']

    num_frames = times.size - 1
    tabs = np.zeros((NUM_STRINGS, num_frames))

    for s in range(NUM_STRINGS):
        string_notes = jam.annotations['note_midi'][s]
        frame_string_pitch = string_notes.to_samples(times[:-1])

        silent = np.array([pitch == [] for pitch in frame_string_pitch])

        frame_string_pitch = np.array(frame_string_pitch)

        frame_string_pitch[silent] = 0

        active_pitches = frame_string_pitch[np.logical_not(silent)].tolist()
        frame_string_pitch[np.logical_not(silent)] = np.array(active_pitches).squeeze()
        frame_string_pitch = frame_string_pitch.astype('float')

        open_string_midi_val = librosa.note_to_midi(TUNING[s])
        frets = frame_string_pitch[np.logical_not(silent)] - open_string_midi_val
        frets = np.round(frets)

        tabs[s, silent] = -1
        tabs[s, np.logical_not(silent)] = frets

    tabs = tabs.astype('int')

    return tabs


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

    # TODO - validate intervals

    # Extract the ground-truth note pitch values into an array
    p_ref = librosa.midi_to_hz(np.array(p_ref))

    return i_ref, p_ref


def write_and_print(file, text, verbose = True, end=''):
    try:
        # Try to write the text to the file (it is assumed to be open)
        file.write(text)
    finally:
        if verbose:
            # Print the text to console
            print(text, end=end)


# TODO - check for NoneType
def write_frames(path, pianoroll, times, places=3):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Open a file at the path with writing permissions
    file = open(path, 'w')

    for i in range(times.size): # For each frame
        # Determine the activations across this frame
        active_pitches = librosa.midi_to_hz(np.where(pianoroll[:, i] != 0)[0] + infer_lowest_note(pianoroll))

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


# TODO - check for NoneType
def write_notes(path, pitches, intervals, places=3):
    os.makedirs(os.path.dirname(path), exist_ok=True)
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


def write_results(results, path, verbose=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Open the file with writing permissions
    results_file = open(path, 'w')
    for type in results.keys():
        write_and_print(results_file, f'-----{type}-----\n', verbose)
        if isinstance(results[type], dict):
            for metric in results[type].keys():
                write_and_print(results_file, f' {metric} : {results[type][metric]}\n', verbose)
        else:
            write_and_print(results_file, f' {type} : {results[type]}\n', verbose)
        write_and_print(results_file, '', verbose, '\n')
    # Close the results file
    results_file.close()


def log_results(results, writer, step=0, metrics=None):
    for type in results.keys():
        if isinstance(results[type], dict):
            for metric in results[type].keys():
                if metrics is None or metric in metrics:
                    writer.add_scalar(f'val/{type}/{metric}', results[type][metric], global_step=step)
        else:
            if metrics is None or type in metrics:
                writer.add_scalar(f'val/{type}', results[type], global_step=step)
