# My imports
from .utils import rms_norm, sort_notes
from .constants import *

# Regular imports
from tqdm import tqdm

import numpy as np
import requests
import zipfile
import librosa
import shutil
import mido
import jams
import os


def load_audio(wav_path, sample_rate=None):
    audio, fs = librosa.load(wav_path, sr=sample_rate)

    # TODO - reassess usefulness of this function
    #audio = librosa.util.normalize(audio)# <- infinity norm
    audio = rms_norm(audio)

    return audio, fs


def load_stacked_notes_jams(jams_path, to_hz=True):
    """
    Load MIDI notes spread across slices (e.g. guitar strings) from JAMS file into a stack.

    Parameters
    ----------
    jams_path : string
      Path to JAMS file to read
    to_hz : bool
      Whether to convert the note pitches to Hertz, as opposed to leaving as MIDI

    Returns
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    """

    # Load the metadata from the jams file
    jam = jams.load(jams_path)

    # Extract all of the midi note annotations
    note_data_slices = jam.annotations[JAMS_NOTE_MIDI]

    # Obtain the number of annotations
    stack_size = len(note_data_slices)

    # Initialize a dictionary to hold the notes
    stacked_notes = dict()

    # Loop through the slices of the stack
    for slice in range(stack_size):
        # Extract the notes pertaining to this slice
        slice_notes = note_data_slices[slice]

        # Initialize lists to hold the pitches and intervals
        pitches, intervals = list(), list()

        # Loop through the notes pertaining to this slice
        for note in slice_notes:
            # Append the note pitch and interval to the respective list
            pitches.append(note.value)
            intervals.append([note.time, note.time + note.duration])

        # Convert the pitch and interval lists to arrays
        pitches, intervals = np.array(pitches), np.array(intervals)

        # Convert pitch to Hertz if specified or leave as MIDI
        if to_hz:
            pitches = librosa.midi_to_hz(pitches)

        # Add the pitch-interval pairs to the stacked notes dictionary under the slice key
        stacked_notes[slice] = sort_notes(pitches, intervals)

    # TODO - validation check?

    return stacked_notes


def load_notes_jams(jams_path, to_hz=True):
    """
    Load all MIDI notes within a JAMS file.

    Parameters
    ----------
    jams_path : string
      Path to JAMS file to read
    to_hz : bool
      Whether to convert the note pitches to Hertz, as opposed to leaving as MIDI

    Returns
    ----------
    pitches : ndarray
      Array of pitches corresponding to notes
    intervals : ndarray
      Array of onset-offset time pairs corresponding to notes
    """

    # First, load the notes into a stack
    stacked_notes = load_stacked_notes_jams(jams_path=jams_path, to_hz=to_hz)

    # Obtain the note pairs from the dictionary values
    note_pairs = list(stacked_notes.values())

    # Extract the pitches and intervals respectively
    pitches = [pair[0] for pair in note_pairs]
    intervals = [pair[1] for pair in note_pairs]

    pitches, intervals = sort_notes(pitches, intervals)

    # TODO - validation check?

    return pitches, intervals


# TODO - clean up or simplify?
def load_jams_guitar_tabs(jams_path, times, tuning):
    jam = jams.load(jams_path)

    # TODO - duration fails at 00_Jazz3-150-C_comp bc of an even division
    # duration = jam.file_metadata['duration']

    note_data = jam.annotations['note_midi']

    num_frames = times.size - 1
    num_strings = len(note_data)
    assert num_strings == len(tuning)
    tabs = np.zeros((num_strings, num_frames))

    for s in range(num_strings):
        string_notes = note_data[s]
        frame_string_pitch = string_notes.to_samples(times[:-1])

        silent = np.array([pitch == [] for pitch in frame_string_pitch])

        frame_string_pitch = np.array(frame_string_pitch)

        frame_string_pitch[silent] = 0

        active_pitches = frame_string_pitch[np.logical_not(silent)].tolist()
        frame_string_pitch[np.logical_not(silent)] = np.array(active_pitches).squeeze()
        frame_string_pitch = frame_string_pitch.astype('float')

        open_string_midi_val = librosa.note_to_midi(tuning[s])
        frets = frame_string_pitch[np.logical_not(silent)] - open_string_midi_val
        frets = np.round(frets)

        tabs[s, silent] = -1
        tabs[s, np.logical_not(silent)] = frets

    tabs = tabs.astype('int')

    return tabs


def load_midi_notes(midi_path):
    """open midi file and return np.array of (onset, offset, note, velocity) rows"""
    midi = mido.MidiFile(midi_path)

    time = 0
    sustain = False
    events = []
    for message in midi:
        time += message.time

        if message.type == 'control_change' and message.control == 64 and (message.value >= 64) != sustain:
            # sustain pedal state has just changed
            sustain = message.value >= 64
            event_type = 'sustain_on' if sustain else 'sustain_off'
            event = dict(index=len(events), time=time, type=event_type, note=None, velocity=0)
            events.append(event)

        if 'note' in message.type:
            # MIDI offsets can be either 'note_off' events or 'note_on' with zero velocity
            velocity = message.velocity if message.type == 'note_on' else 0
            event = dict(index=len(events), time=time, type='note', note=message.note, velocity=velocity, sustain=sustain)
            events.append(event)

    notes = []
    for i, onset in enumerate(events):
        if onset['velocity'] == 0:
            continue

        # find the next note_off message
        offset = next(n for n in events[i + 1:] if n['note'] == onset['note'] or n is events[-1])

        if offset['sustain'] and offset is not events[-1]:
            # if the sustain pedal is active at offset, find when the sustain ends
            offset = next(n for n in events[offset['index'] + 1:] if n['type'] == 'sustain_off' or n is events[-1])

        note = (onset['time'], offset['time'], onset['note'], onset['velocity'])
        notes.append(note)

    return np.array(notes)


def write_and_print(file, text, verbose = True, end=''):
    try:
        # Try to write the text to the file (it is assumed to be open)
        file.write(text)
    finally:
        if verbose:
            # Print the text to console
            print(text, end=end)


# TODO - make sure it works for NoneType or empty
def write_pitch(path, pitch, times, low, places=3):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Open a file at the path with writing permissions
    file = open(path, 'w')

    # TODO - can this be done without a loop?
    for i in range(times.size): # For each frame
        # Determine the activations across this frame
        # TODO - change this to Hz
        active_pitches = librosa.midi_to_hz(np.where(pitch[:, i] != 0)[0] + low)

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


def write_pitch_multi(log_dir, multi, times, low, labels=None, places=3):
    multi_num = multi.shape[0]

    for i in range(multi_num):
        if labels is None:
            path = os.path.join(log_dir, i, '.txt')
        else:
            path = os.path.join(log_dir, labels[i], '.txt')

        write_pitch(path, multi[i], times, low, places)


# TODO - check for NoneType
def write_notes(path, pitches, intervals, places=3):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Open a file at the path with writing permissions
    file = open(path, 'w')

    # TODO - can this be done without a loop?
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


def write_notes_multi(log_dir, notes_multi, labels=None, places=3):
    multi_num = len(notes_multi)

    for i in range(multi_num):
        if labels is None:
            path = os.path.join(log_dir, i, '.txt')
        else:
            path = os.path.join(log_dir, labels[i], '.txt')

        pitches, intervals = notes_multi[i]
        write_notes(path, pitches, intervals, places)


def stream_url_resource(url, save_path, chunk_size=1024):
    """
    Download a file at a URL by streaming it.

    Parameters
    ----------
    url : string
      URL pointing to the file
    save_path : string
      Path to the save location (including the file name)
    chunk_size : int
      Number of bytes to download at a time
    """

    # Create an HTTP GET request
    r = requests.get(url, stream=True)

    # Determine the total number of bytes to be downloaded
    total_length = int(r.headers.get('content-length'))

    # Open the target file in write mode
    with open(save_path, 'wb') as file:
        # Iteratively download chunks of the file,
        # displaying a progress bar in the console
        for chunk in tqdm(r.iter_content(chunk_size=chunk_size),
                          total=int(total_length/chunk_size+1)):
            # If a chunk was successfully downloaded,
            if chunk:
                # Write the chunk to the file
                file.write(chunk)


def unzip_and_remove(zip_path, target=None):
    """
    Unzip a zip file and remove it.

    Parameters
    ----------
    zip_path : string
      Path to the zip file
    target : string or None
      Directory to extract the zip file contents into
    """

    print(f'Unzipping {os.path.basename(zip_path)}')

    # Default the save location as the same directory as the zip file
    if target is None:
        target = os.path.dirname(zip_path)

    # Open the zip file in read mode
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract the contents of the zip file into the target directory
        zip_ref.extractall(target)

    # Delete the zip file
    os.remove(zip_path)


def change_base_dir(new_dir, old_dir):
    """
    Change the top-level directory from the path chain of each file.

    Parameters
    ----------
    new_dir : string
      New top-level directory
    old_dir : string
      Old top-level directory
    """

    # Loop through all contents of the old directory
    for content in os.listdir(old_dir):
        # Construct the old path to the contents
        old_path = os.path.join(old_dir, content)
        # Construct the new path to the contents
        new_path = os.path.join(new_dir, content)
        # Move all files and subdirectories recursively
        shutil.move(old_path, new_path)

    # Remove the (now empty) old top-level directory
    os.rmdir(old_dir)
