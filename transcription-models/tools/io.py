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


def load_jams_guitar_tabs(jams_path, hop_length, num_frames, sample_rate):
    jam = jams.load(jams_path)

    # TODO - duration fails at 00_Jazz3-150-C_comp bc of an even division
    # duration = jam.file_metadata['duration']

    tabs = np.zeros((NUM_STRINGS, NUM_FRETS + 2, num_frames))

    frame_times = librosa.frames_to_time(range(num_frames), sample_rate, hop_length)

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


def write_and_print(file, text, verbose = True, end=''):
    try:
        # Try to write the text to the file (it is assumed to be open)
        file.write(text)
    finally:
        if verbose:
            # Print the text to console
            print(text, end=end)


def write_frames(path, h_len, sample_rate, pianoroll, places = 3):
    # Open a file at the path with writing permissions
    file = open(path, 'w')

    # Calculate the times corresponding to each frame prediction
    # TODO - window length is ambiguous - remove? - also check librosa func
    #times = (0.5 * w_len + np.arange(pitches.shape[0]) * h_len) / SAMPLE_RATE
    times = (np.arange(pianoroll.shape[1]) * h_len) / sample_rate

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
