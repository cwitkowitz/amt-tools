# My imports
import amt_models.tools.utils as utils
import amt_models.tools.constants as constants

# Regular imports
from mir_eval.multipitch import resample_multipitch
from tqdm import tqdm

import numpy as np
import requests
import zipfile
import librosa
import shutil
import mido
import jams
import os


##################################################
# INPUT                                          #
##################################################


def load_normalize_audio(wav_path, fs=None, norm=-1):
    """
    Load audio from a file and normalize it.

    Parameters
    ----------
    wav_path : string
      Path to audio file to read
    fs : int or float or None (optional)
      Desired sampling rate
    norm : float
      Type of normalization to perform
      -1 - root-mean-square
      other - see librosa

    Returns
    ----------
    audio : ndarray (N)
      Mono-channel audio read from file
      N - number of samples in audio
    fs : int
      Audio sampling rate
    """

    # Load the audio using librosa
    audio, fs = librosa.load(wav_path, sr=fs, mono=True)

    if norm == -1:
        # Perform root-mean-square normalization
        audio = utils.rms_norm(audio)
    else:
        # Normalize the audio using librosa
        audio = librosa.util.normalize(audio, norm)

    return audio, fs


def load_stacked_notes_jams(jams_path):
    """
    Load MIDI notes spread across slices (e.g. guitar strings) from JAMS file into a dictionary.

    Parameters
    ----------
    jams_path : string
      Path to JAMS file to read

    Returns
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    """

    # Load the metadata from the jams file
    jam = jams.load(jams_path)

    # Extract all of the midi note annotations
    note_data_slices = jam.annotations[constants.JAMS_NOTE_MIDI]

    # Obtain the number of annotations
    stack_size = len(note_data_slices)

    # Initialize a dictionary to hold the notes
    stacked_notes = dict()

    # Loop through the slices of the stack
    for slc in range(stack_size):
        # Extract the notes pertaining to this slice
        slice_notes = note_data_slices[slc]

        # Initialize lists to hold the pitches and intervals
        pitches, intervals = list(), list()

        # Loop through the notes pertaining to this slice
        for note in slice_notes:
            # Append the note pitch and interval to the respective list
            pitches.append(note.value)
            intervals.append([note.time, note.time + note.duration])

        # Convert the pitch and interval lists to arrays
        pitches, intervals = np.array(pitches), np.array(intervals)

        # Add the pitch-interval pairs to the stacked notes dictionary under the slice key
        stacked_notes.update(utils.notes_to_stacked_notes(pitches, intervals, slc))

    return stacked_notes


def load_notes_jams(jams_path):
    """
    Load all MIDI notes within a JAMS file.

    Parameters
    ----------
    jams_path : string
      Path to JAMS file to read

    Returns
    ----------
    pitches : ndarray (N)
      Array of pitches corresponding to notes
      N - number of notes
    intervals : ndarray (N x 2)
      Array of onset-offset time pairs corresponding to notes
      N - number of notes
    """

    # First, load the notes into a stack
    stacked_notes = load_stacked_notes_jams(jams_path=jams_path)

    pitches, intervals = utils.stacked_notes_to_notes(stacked_notes)

    return pitches, intervals


def load_stacked_pitch_list_jams(jams_path, times=None):
    """
    Load pitch lists spread across slices (e.g. guitar strings) from JAMS file into a dictionary.

    Parameters
    ----------
    jams_path : string
      Path to JAMS file to read
    times : ndarray or None (optional) (N)
      Time in seconds of beginning of each frame
      N - number of times samples

    Returns
    ----------
    stacked_pitch_list : dict
      Dictionary containing (slice -> (times, pitch_list)) pairs
    """

    # Load the metadata from the jams file
    jam = jams.load(jams_path)

    # Extract all of the pitch annotations
    pitch_data_slices = jam.annotations[constants.JAMS_PITCH_HZ]

    # Obtain the number of annotations
    stack_size = len(pitch_data_slices)

    # Initialize a dictionary to hold the pitch lists
    stacked_pitch_list = dict()

    # Loop through the slices of the stack
    for slc in range(stack_size):
        # Extract the pitch list pertaining to this slice
        slice_pitches = pitch_data_slices[slc]

        # Initialize an array/list to hold the times/frequencies associated with each observation
        entry_times, slice_pitch_list = np.empty(0), list()

        # Loop through the pitch observations pertaining to this slice
        for pitch in slice_pitches:
            # Extract the pitch
            freq = np.array([pitch.value['frequency']])

            # Don't keep track of zero-frequencies
            if np.sum(freq) == 0:
                freq = np.empty(0)

            # Append the observation time
            entry_times = np.append(entry_times, pitch.time)
            # Append the frequency
            slice_pitch_list.append(freq)

        if times is not None:
            # Sort the pitch list before resampling just in case it is not already sorted
            entry_times, slice_pitch_list = utils.sort_pitch_list(entry_times, slice_pitch_list)

            # Resample the observation times if new times are specified
            gap_tolerance = 2 * np.min(times[1:] - times[:-1])
            slice_pitch_list = resample_multipitch(entry_times, slice_pitch_list, times, gap_tolerance)
            # Overwrite the entry times with the specified times
            entry_times = times

        # Add the pitch list to the stacked pitch list dictionary under the slice key
        stacked_pitch_list.update(utils.pitch_list_to_stacked_pitch_list(entry_times,
                                                                         slice_pitch_list,
                                                                         slc))

    return stacked_pitch_list


def load_pitch_list_jams(jams_path, times):
    """
    Load pitch list from JAMS file.

    Parameters
    ----------
    jams_path : string
      Path to JAMS file to read
    times : ndarray or None (optional) (N)
      Time in seconds of beginning of each frame
      N - number of times samples

    Returns
    ----------
    times : ndarray (N)
      Time in seconds of beginning of each frame, sorted by time
      N - number of time samples (frames)
    pitch_list : list of ndarray (N x [...])
      Array of pitches corresponding to notes
      N - number of pitch observations (frames), sorted by time
    """

    # Load the pitch lists into a stack
    stacked_pitch_list = load_stacked_pitch_list_jams(jams_path, times)

    # Convert them to a single pitch list
    times, pitch_list = utils.stacked_pitch_list_to_pitch_list(stacked_pitch_list)

    return times, pitch_list


def load_notes_midi(midi_path):
    """
    Load all MIDI notes from a MIDI file, keeping track of sustain pedal activity.
    TODO - make sustain pedal stuff optional?

    Parameters
    ----------
    midi_path : string
      Path to MIDI file to read

    Returns
    ----------
    batched_notes : ndarray (N x 4)
      Array of note pitches, intervals, and velocities by row
      N - number of notes
    """

    # Open the MIDI file
    midi = mido.MidiFile(midi_path)

    # Initialize a counter for the time
    time = 0

    # Keep track of sustain pedal activity
    sustain_status = False

    # Initialize an empty list to store MIDI events
    events = []

    # Loop through MIDI messages, keeping track of only those pertaining to notes and sustain pedal activity
    for message in midi:
        # Increment the time
        time += message.time

        # Check if the message constitutes a control change event
        if message.type == constants.MIDI_CONTROL_CHANGE:
            # Whether the message constitutes a sustain control event
            sustain_control = message.control == constants.MIDI_SUSTAIN_CONTROL_NUM

            # Value >= 64 means SUSTAIN_ON and value < 64 means SUSTAIN OFF
            sustain_on = message.value >= constants.MIDI_SUSTAIN_CONTROL_NUM

            # Whether the current status of the sustain pedal was actually changed
            sustain_change = sustain_on != sustain_status

            # Check if the above two conditions were met (sustain control and status change)
            if sustain_control and sustain_change:
                # Update the status of the sustain pedal (on/off)
                sustain_status = sustain_on
                # Determine which event occurred (SUSTAIN_ON or SUSTAIN_OFF)
                event_type = constants.MIDI_SUSTAIN_ON if sustain_status else constants.MIDI_SUSTAIN_OFF

                # Create a new event detailing the sustain pedal activity
                event = dict(index=len(events),
                             time=time,
                             type=event_type,
                             note=None,
                             velocity=0)
                # Add the sustain pedal event to the MIDI event list
                events.append(event)

        # Check if the message constitutes a note event (NOTE_ON or NOTE_OFF)
        if 'note' in message.type:
            # MIDI offsets can be either NOTE_OFF events or NOTE_ON with zero velocity
            velocity = message.velocity if message.type == constants.MIDI_NOTE_ON else 0

            # Create a new event detailing the note and current sustain pedal state
            event = dict(index=len(events),
                         time=time,
                         type='note',
                         note=message.note,
                         velocity=velocity,
                         sustain=sustain_status)
            # Add the note event to the MIDI event list
            events.append(event)

    # Initialize an empty list to store notes
    notes = list()

    # Loop through all of the documented MIDI events
    for i, onset in enumerate(events):
        # Ignore anything but note onset events (note offsets and sustain activity ignored)
        if onset['velocity'] == 0:
            continue

        # Determine where the corresponding offset occurs
        # by finding the next note event with the same pitch,
        # clipping at the final frame if no correspondence is found
        offset = next(n for n in events[i + 1:] if n['note'] == onset['note'] or n is events[-1])

        # Check if the sustain pedal is on when the note offset occurs
        if offset['sustain'] and offset is not events[-1]:
            # If so, offset is when sustain ends or another note event of same pitch occurs
            offset = next(n for n in events[offset['index'] + 1:] if n['type'] == constants.MIDI_SUSTAIN_OFF or
                          n['note'] == onset['note'] or n is events[-1])

        # Create a new note entry and append it to the list of notes
        notes.append([onset['time'], offset['time'], onset['note'], onset['velocity']])

    # Package the notes into an array
    notes = np.array(notes)

    return notes


##################################################
# OUTPUT                                         #
##################################################


def write_and_print(file, text, verbose = True, end=''):
    try:
        # Try to write the text to the file (it is assumed to be open)
        file.write(text)
    finally:
        if verbose:
            # Print the text to console
            print(text, end=end)


# TODO - make sure it works for empty
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
            path = os.path.join(log_dir, i, constants.TXT_EXT)
        else:
            path = os.path.join(log_dir, labels[i], constants.TXT_EXT)

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
            path = os.path.join(log_dir, i, constants.TXT_EXT)
        else:
            path = os.path.join(log_dir, labels[i], constants.TXT_EXT)

        pitches, intervals = notes_multi[i]
        write_notes(path, pitches, intervals, places)


##################################################
# USEFUL FILE MANAGEMENT                         #
##################################################


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
