# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from . import constants

# Regular imports
from datetime import datetime
from copy import deepcopy
from scipy import signal

import numpy as np
import warnings
import librosa
import random
import torch
import scipy
import time


# TODO - torch Tensor compatibility
# TODO - try to ensure these won't break if extra dimensions (e.g. batch) are included
# TODO - make sure there are no hard assignments (make return copies instead of original where necessary)

__all__ = [
    'notes_to_batched_notes',
    'cat_batched_notes',
    'filter_batched_note_repeats',
    'transpose_batched_notes',
    'stacked_notes_to_batched_notes',
    'batched_notes_to_hz',
    'batched_notes_to_midi',
    'slice_batched_notes',
    'multi_pitch_to_notes',
    'batched_notes_to_notes',
    'stacked_notes_to_notes',
    'notes_to_hz',
    'notes_to_midi',
    'offset_notes',
    'detect_overlap_notes',
    'filter_notes',
    'notes_to_stacked_notes',
    'batched_notes_to_stacked_notes',
    'stacked_notes_to_hz',
    'stacked_notes_to_midi',
    'cat_stacked_notes',
    'filter_stacked_note_repeats',
    'stacked_notes_to_frets',
    'find_pitch_bounds_stacked_notes',
    'stacked_pitch_list_to_pitch_list',
    'multi_pitch_to_pitch_list',
    'pitch_list_to_hz',
    'pitch_list_to_midi',
    'slice_pitch_list',
    'cat_pitch_list',
    'unroll_pitch_list',
    'clean_pitch_list',
    'pack_pitch_list',
    'unpack_pitch_list',
    'get_active_pitch_count',
    'contains_empties_pitch_list',
    'detect_overlap_pitch_list',
    'filter_pitch_list',
    'pitch_list_to_stacked_pitch_list',
    'stacked_multi_pitch_to_stacked_pitch_list',
    'stacked_pitch_list_to_hz',
    'stacked_pitch_list_to_midi',
    'slice_stacked_pitch_list',
    'cat_stacked_pitch_list',
    'notes_to_multi_pitch',
    'pitch_list_to_multi_pitch',
    'stacked_multi_pitch_to_multi_pitch',
    'logistic_to_stacked_multi_pitch',
    'stacked_notes_to_stacked_multi_pitch',
    'stacked_pitch_list_to_stacked_multi_pitch',
    'multi_pitch_to_stacked_multi_pitch',
    'tablature_to_stacked_multi_pitch',
    'stacked_pitch_list_to_tablature',
    'stacked_multi_pitch_to_tablature',
    'logistic_to_tablature',
    'stacked_multi_pitch_to_logistic',
    'tablature_to_logistic',
    'notes_to_onsets',
    'multi_pitch_to_onsets',
    'stacked_notes_to_stacked_onsets',
    'stacked_multi_pitch_to_stacked_onsets',
    'notes_to_offsets',
    'multi_pitch_to_offsets',
    'stacked_notes_to_stacked_offsets',
    'stacked_multi_pitch_to_stacked_offsets',
    'sort_batched_notes',
    'sort_notes',
    'sort_pitch_list',
    'rms_norm',
    'blur_activations',
    'normalize_activations',
    'threshold_activations',
    'framify_activations',
    'inhibit_activations',
    'remove_activation_blips',
    'interpolate_gaps',
    'get_resample_idcs',
    'seed_everything',
    'estimate_hop_length',
    'time_series_to_uniform',
    'get_frame_times',
    'apply_func_stacked_representation',
    'pack_stacked_representation',
    'unpack_stacked_representation',
    'tensor_to_array',
    'array_to_tensor',
    'save_dict_npz',
    'load_dict_npz',
    'dict_to_dtype',
    'dict_to_device',
    'dict_to_array',
    'dict_to_tensor',
    'dict_squeeze',
    'dict_unsqueeze',
    'dict_append',
    'dict_detach',
    'unpack_dict',
    'query_dict',
    'get_tag',
    'slice_track',
    'get_current_time',
    'print_time',
    'compute_time_difference'
]


##################################################
# TO BATCH-FRIENDLY NOTES                        #
##################################################

def notes_to_batched_notes(pitches, intervals):
    """
    Convert loose note groups into batch-friendly storage.

    Parameters
    ----------
    pitches : ndarray (N)
      Array of pitches corresponding to notes
      N - number of notes
    intervals : ndarray (N x 2)
      Array of onset-offset time pairs corresponding to notes
      N - number of notes

    Returns
    ----------
    batched_notes : ndarray (N x 3)
      Array of note intervals and pitches by row
      N - number of notes
    """

    # Default the batched notes to an empty array of the correct shape
    batched_notes = np.empty([0, 3])

    if len(pitches) > 0:
        # Add an extra dimension to the pitches to match dimensionality of intervals
        pitches = np.expand_dims(pitches, axis=-1)
        # Concatenate the loose arrays to obtain ndarray([[onset, offset, pitch]])
        batched_notes = np.concatenate((intervals, pitches), axis=-1)

    return batched_notes


def cat_batched_notes(batched_notes, new_batched_notes):
    """
    Concatenate two collections of batched notes.

    Parameters
    ----------
    batched_notes : ndarray (N x 3)
      Array of note intervals and pitches by row or column
      N - number of notes
    new_batched_notes : ndarray (N x 3)
      Same as batched_notes

    Returns
    ----------
    batched_notes : ndarray (N x 3)
      Array of note intervals and pitches by row or column
      N - number of notes
    """

    # Concatenate along the first axis
    batched_notes = np.concatenate((batched_notes, new_batched_notes), axis=0)

    return batched_notes


def filter_batched_note_repeats(batched_notes):
    """
    Remove any note duplicates, where a duplicate is defined as
    an entry with the same pitch and onset time. If there are
    duplicates, we keep the entry with the longest duration.

    Parameters
    ----------
    batched_notes : ndarray (N x 3)
      Array of note intervals and pitches by row or column
      N - number of notes

    Returns
    ----------
    batched_notes : ndarray (L x 3)
      Array of note intervals and pitches by row or column
      L - number of notes
    """

    # Sort the batched notes, so the longest duration will always be chosen
    batched_notes = np.flip(sort_batched_notes(batched_notes), axis=0)

    # Determine the pitches and onsets represented by the batched_notes
    pitches_onsets = np.roll(batched_notes, shift=1, axis=-1)[:, :2]

    # Determine which note entries should be kept
    keep_indices = np.unique(pitches_onsets, return_index=True, axis=0)[-1]

    # Remove duplicate note entries
    batched_notes = batched_notes[keep_indices]

    return batched_notes


def transpose_batched_notes(batched_notes):
    """
    Switch the axes of batched notes.

    Parameters
    ----------
    batched_notes : ndarray (3 x N) or (N x 3)
      Array of note intervals and pitches by row or column
      N - number of notes

    Returns
    ----------
    batched_notes : ndarray (N x 3) or (3 x N)
      Array of note intervals and pitches by row or column
      N - number of notes
    """

    # Transpose the last two axes
    batched_notes = np.transpose(batched_notes, (-1, -2))

    return batched_notes


def stacked_notes_to_batched_notes(stacked_notes, transposed=False):
    """
    Convert a dictionary of stacked notes into a single representation.

    Parameters
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> batched_notes) pairs
    transposed : bool
      Whether the axes of the batched notes were transposed before they were added

    Returns
    ----------
    batched_notes : ndarray (N x 3)
      Array of note intervals and pitches by row
      N - number of notes
    """

    # Obtain the batched note entries from the dictionary values
    entries = list(stacked_notes.values())

    # Concatenate all groups of batched notes
    batched_notes = np.concatenate([entry for entry in entries], axis=int(transposed))

    return batched_notes


def batched_notes_to_hz(batched_notes):
    """
    Convert batched notes from MIDI to Hertz.

    Parameters
    ----------
    batched_notes : ndarray (N x 3)
      Array of note intervals and MIDI pitches by row
      N - number of notes

    Returns
    ----------
    batched_notes : ndarray (N x 3)
      Array of note intervals and Hertz pitches by row
      N - number of notes
    """

    # Convert pitch column to Hertz
    batched_notes[..., 2] = librosa.midi_to_hz(batched_notes[..., 2])

    return batched_notes


def batched_notes_to_midi(batched_notes):
    """
    Convert batched notes from Hertz to MIDI.

    Parameters
    ----------
    batched_notes : ndarray (N x 3)
      Array of note intervals and Hertz pitches by row
      N - number of notes

    Returns
    ----------
    batched_notes : ndarray (N x 3)
      Array of note intervals and MIDI pitches by row
      N - number of notes
    """

    # Convert pitch column to MIDI
    batched_notes[..., 2] = librosa.hz_to_midi(batched_notes[..., 2])

    return batched_notes


def slice_batched_notes(batched_notes, start_time, stop_time, relative_times=False):
    """
    Remove note entries occurring outside of time window.

    Parameters
    ----------
    batched_notes : ndarray (N x 3)
      Array of note intervals and pitches by row
      N - number of notes
    start_time : float
      Beginning of time window
    stop_time : float
      End of time window
    relative_times : bool
      Whether onsets/offsets should be relative to time origin

    Returns
    ----------
    batched_notes : ndarray (N x 3)
      Array of note intervals and pitches by row
      N - number of notes
    """

    # Remove notes with offsets before the slice start time
    batched_notes = batched_notes[batched_notes[:, 1] > start_time]

    # Remove notes with onsets after the slice stop time
    batched_notes = batched_notes[batched_notes[:, 0] <= stop_time]

    # Clip onsets at the slice start time
    batched_notes[:, 0] = np.maximum(batched_notes[:, 0], start_time)

    # Clip offsets at the slice stop time
    batched_notes[:, 1] = np.minimum(batched_notes[:, 1], stop_time)

    if relative_times:
        # Adjust onset/offset times
        batched_notes[:, :2] -= start_time

    return batched_notes


##################################################
# TO NOTES                                       #
##################################################


def multi_pitch_to_notes(multi_pitch, times, profile, onsets=None, offsets=None):
    """
    Decode a multi pitch array into loose MIDI note groups.

    Parameters
    ----------
    multi_pitch : ndarray (F x T)
      Discrete pitch activation map
      F - number of discrete pitches
      T - number of frames
    times : ndarray (N)
      Time in seconds of beginning of each frame
      N - number of time samples (frames)
    profile : InstrumentProfile (instrument.py)
      Instrument profile detailing experimental setup
    onsets : ndarray (F x T) or None (Optional)
      Where to start considering notes "active"
      F - number of discrete pitches
      T - number of frames
    offsets : ndarray (F x T) or None (Optional)
      Where to stop considering notes "active" - currently unused
      F - number of discrete pitches
      T - number of frames

    Returns
    ----------
    pitches : ndarray (N)
      Array of pitches corresponding to notes in MIDI format
      N - number of notes
    intervals : ndarray (N x 2)
      Array of onset-offset time pairs corresponding to notes
      N - number of notes
    """

    if onsets is None:
        # Default the onsets if they were not provided
        onsets = multi_pitch_to_onsets(multi_pitch)

    # Make sure all onsets have corresponding pitch activations
    multi_pitch = np.logical_or(onsets, multi_pitch).astype(constants.FLOAT32)

    # Turn onset activations into impulses at starting frame
    onsets = multi_pitch_to_onsets(onsets)

    # Determine the total number of frames
    num_frames = multi_pitch.shape[-1]

    # Estimate the duration of the track (for bounding note offsets)
    times = np.append(times, times[-1] + estimate_hop_length(times))

    # Create empty lists for note pitches and their time intervals
    pitches, intervals = list(), list()

    # Determine the pitch and frame indices where notes begin
    pitch_idcs, frame_idcs = onsets.nonzero()

    # Loop through note beginnings
    for pitch, frame in zip(pitch_idcs, frame_idcs):
        # Mark onset and start offset counter
        onset, offset = frame, frame + 1

        # Increment the offset counter until one of the following occurs:
        #  1. There are no more frames
        #  2. Pitch is no longer active in the multi pitch array
        #  3. A new onset occurs involving the current pitch
        while True:
            # There are no more frames to count
            maxed_out = offset == num_frames

            if maxed_out:
                # Stop looping
                break

            # There is an activation for the pitch at the next frame
            active_pitch = multi_pitch[pitch, offset]

            if not active_pitch:
                # Stop looping
                break

            # There is an onset for the pitch at the next frame
            new_onset = onsets[pitch, offset]

            if new_onset:
                # Stop looping
                break

            # Include the offset counter
            offset += 1

        # Add the frequency to the list
        pitches.append(pitch + profile.low)

        # Add the interval to the list
        intervals.append([times[onset], times[offset]])

    # Convert the lists to numpy arrays
    pitches, intervals = np.array(pitches), np.array(intervals)

    # Sort notes by onset just for the purpose of being neat
    pitches, intervals = sort_notes(pitches, intervals)

    return pitches, intervals


def batched_notes_to_notes(batched_notes):
    """
    Convert batch-friendly notes into loose note groups.

    Parameters
    ----------
    batched_notes : ndarray (N x 3)
      Array of note intervals and pitches by row
      N - number of notes

    Returns
    ----------
    pitches : ndarray (N)
      Array of pitches corresponding to notes
      N - number of notes
    intervals : ndarray (N x 2)
      Array of onset-offset time pairs corresponding to notes
      N - number of notes
    """

    # Split along the final dimension into the loose groups
    pitches, intervals = batched_notes[..., 2], batched_notes[:, :2]

    return pitches, intervals


def stacked_notes_to_notes(stacked_notes, sort_by=0):
    """
    Convert a dictionary of stacked notes into a single representation.

    Parameters
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    sort_by : int or None (Optional)
      Index to sort notes by
      0 - onset | 1 - offset | 2 - pitch

    Returns
    ----------
    pitches : ndarray (N)
      Array of pitches corresponding to notes
      N - number of notes
    intervals : ndarray (N x 2)
      Array of onset-offset time pairs corresponding to notes
      N - number of notes
    """

    # Obtain the note pairs from the dictionary values
    note_pairs = list(stacked_notes.values())

    # Extract the pitches and intervals respectively
    pitches = np.concatenate([pair[0] for pair in note_pairs])
    intervals = np.concatenate([pair[1] for pair in note_pairs])

    if sort_by is not None:
        # Sort the notes by the specified attribute
        pitches, intervals = sort_notes(pitches, intervals, by=sort_by)

    return pitches, intervals


def notes_to_hz(pitches):
    """
    Convert note pitches from MIDI to Hertz.
    Array of corresponding intervals does not change and is
    assumed to be managed outside of the function.

    Parameters
    ----------
    pitches : ndarray (N)
      Array of MIDI pitches corresponding to notes
      N - number of notes

    Returns
    ----------
    pitches : ndarray (N)
      Array of Hertz pitches corresponding to notes
      N - number of notes
    """

    # Convert to Hertz
    pitches = librosa.midi_to_hz(pitches)

    return pitches


def notes_to_midi(pitches):
    """
    Convert note pitches from Hertz to MIDI.
    Array of corresponding intervals does not change and is
    assumed to be managed outside of the function.

    Parameters
    ----------
    pitches : ndarray (N)
      Array of Hertz pitches corresponding to notes
      N - number of notes

    Returns
    ----------
    pitches : ndarray (N)
      Array of MIDI pitches corresponding to notes
      N - number of notes
    """

    # Convert to MIDI
    pitches = librosa.hz_to_midi(pitches)

    return pitches


def offset_notes(pitches, intervals, semitones):
    """
    Add a semitone offset to note groups.

    Parameters
    ----------
    pitches : ndarray (N)
      Array of pitches corresponding to notes
      N - number of notes
    intervals : ndarray (N x 2)
      Array of onset-offset time pairs corresponding to notes
      N - number of notes
    semitones : float
      Number of semitones by which to offset note pitches

    Returns
    ----------
    pitches : ndarray (N)
      Same as input with added offset
    intervals : ndarray (N x 2)
      Same as input
    """

    # Add the offset to the pitches
    pitches += semitones

    return pitches, intervals


def detect_overlap_notes(intervals, decimals=3):
    """
    Determine if a set of intervals contains any overlap.

    Parameters
    ----------
    intervals : ndarray (N x 2)
      Array of onset-offset time pairs
      N - number of notes
    decimals : int (Optional - millisecond by default)
      Decimal resolution for timing comparison

    Returns
    ----------
    overlap : bool
      Whether any intervals overlap
    """

    # Make sure the intervals are sorted by onset (abusing this function slightly)
    intervals = sort_batched_notes(intervals, by=0)
    # Check if any onsets occur before the offset of a previous interval
    overlap = np.sum(np.round(np.diff(intervals).flatten(), decimals) < 0) > 0

    return overlap


def filter_notes(pitches, intervals, profile=None, min_time=-np.inf, max_time=np.inf, suppress_warnings=True):
    """
    Remove notes with nominal pitch outside the supported range of an instrument
    profile and notes with intervals entirely outside of specified boundaries.

    Parameters
    ----------
    pitches : ndarray (N)
      Array of pitches corresponding to notes
      N - number of notes
    intervals : ndarray (N x 2)
      Array of onset-offset time pairs corresponding to notes
      N - number of notes
    profile : InstrumentProfile or None (Optional)
      Instrument profile detailing experimental setup
    min_time : float (Optional)
      Note offsets must occur at or after this time to be considered valid
    max_time : float (Optional)
      Note onsets must occur at or before this time to be considered valid
    suppress_warnings : bool
      Whether to ignore warning messages

    Returns
    ----------
    pitches : ndarray (K)
      Pitches corresponding to filtered notes
      K - number of within-bounds notes
    intervals : ndarray (K x 2)
      Intervals corresponding to filtered notes
      K - number of within-bounds notes
    """

    # Round pitches to the nearest semitone
    pitches_r = np.round(pitches)

    if profile is not None:
        # Check for notes with out-of-bounds pitches (w.r.t. specified instrument profile)
        in_bounds_pitch = np.logical_and((pitches_r >= profile.low), (pitches_r <= profile.high))

        if np.sum(np.logical_not(in_bounds_pitch)) and not suppress_warnings:
            # Print a warning message if notes were ignored
            warnings.warn('Ignoring notes with nominal pitch exceeding ' +
                          'supported boundaries.', category=RuntimeWarning)

    # Check for notes with onsets occurring after specified time maximum
    in_bounds_interval_on = (intervals[:, 0] <= max_time)

    if np.sum(np.logical_not(in_bounds_interval_on)) and not suppress_warnings:
        # Print a warning message if notes were ignored
        warnings.warn('Ignoring notes with onsets occurring after ' +
                      'specified time maximum.', category=RuntimeWarning)

    # Check for notes with offsets occurring before specified time minimum
    in_bounds_interval_off = (intervals[:, 1] >= min_time)

    if np.sum(np.logical_not(in_bounds_interval_off)) and not suppress_warnings:
        # Print a warning message if notes were ignored
        warnings.warn('Ignoring notes with offsets occurring before ' +
                      'specified time minimum.', category=RuntimeWarning)

    # Combine valid indices from onsets/offsets checks
    valid_idcs = np.logical_and(in_bounds_interval_on, in_bounds_interval_off)

    if profile is not None:
        # Combine pitches/interval checks to determine if and where there are out-of-bounds notes
        valid_idcs = np.logical_and(valid_idcs, in_bounds_pitch)

    # Remove any invalid notes (out-of-bounds w.r.t. interval or pitch)
    pitches, intervals = pitches[valid_idcs], intervals[valid_idcs]

    return pitches, intervals


##################################################
# TO STACKED NOTES                               #
##################################################


def notes_to_stacked_notes(pitches, intervals, key=0):
    """
    Convert a collection of notes into a dictionary of stacked notes.

    Parameters
    ----------
    pitches : ndarray (N)
      Array of pitches corresponding to notes
      N - number of notes
    intervals : ndarray (N x 2)
      Array of onset-offset time pairs corresponding to notes
      N - number of notes
    key : object
      Slice key to use

    Returns
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    """

    # Initialize a dictionary to hold the notes
    stacked_notes = dict()

    # Add the pitch-interval pairs to the stacked notes dictionary under the slice key
    stacked_notes[key] = sort_notes(pitches, intervals)

    return stacked_notes


def batched_notes_to_stacked_notes(batched_notes, transposed=False, i=0):
    """
    Convert a collection of (batched) notes into a dictionary of stacked notes.

    Parameters
    ----------
    batched_notes : ndarray (N x 3)
      Array of note intervals and pitches by row
      N - number of notes
    transposed : bool
      Whether to switch the axes of the batched notes before adding them
    i : int
      Slice key to use

    Returns
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> batched_notes)
    """

    # Initialize a dictionary to hold the notes
    stacked_notes = dict()

    batched_notes = sort_batched_notes(batched_notes)

    if transposed:
        # Switch the axes
        batched_notes = transpose_batched_notes(batched_notes)

    # Add the batched notes to the stacked notes dictionary under the slice key
    stacked_notes[i] = batched_notes

    return stacked_notes


def stacked_notes_to_hz(stacked_notes):
    """
    Convert stacked notes from MIDI to Hertz.

    Parameters
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches (MIDI), intervals)) pairs

    Returns
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches (Hertz), intervals)) pairs
    """

    # Make a copy of the stacked notes for conversion
    stacked_notes = deepcopy(stacked_notes)

    # Loop through the stack of notes
    for slc in stacked_notes.keys():
        # Get the pitches from the slice
        pitches, intervals = stacked_notes[slc]
        # Convert the pitches to Hertz
        pitches = notes_to_hz(pitches)
        # Add converted slice back to stack
        stacked_notes[slc] = pitches, intervals

    return stacked_notes


def stacked_notes_to_midi(stacked_notes):
    """
    Convert stacked notes from Hertz to MIDI.

    Parameters
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches (Hertz), intervals)) pairs

    Returns
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches (MIDI), intervals)) pairs
    """

    # Make a copy of the stacked notes for conversion
    stacked_notes = deepcopy(stacked_notes)

    # Loop through the stack of notes
    for slc in stacked_notes.keys():
        # Get the pitches from the slice
        pitches, intervals = stacked_notes[slc]
        # Convert the pitches to MIDI
        pitches = notes_to_midi(pitches)
        # Add converted slice back to stack
        stacked_notes[slc] = pitches, intervals

    return stacked_notes


def cat_stacked_notes(stacked_notes, new_stacked_notes):
    """
    Concatenate two collections of stacked notes.

    Parameters
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    new_stacked_notes : dict
      Same as stacked_notes
      Note: must also have same number of slices

    Returns
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    """

    # Make a copy of the stacked pitch lists for concatenation
    stacked_notes = deepcopy(stacked_notes)

    # Loop through the stack of pitch lists
    for slc in stacked_notes.keys():
        # Convert the notes to batched notes
        batched_notes = notes_to_batched_notes(*stacked_notes[slc])
        new_batched_notes = notes_to_batched_notes(*new_stacked_notes[slc])
        # Concatenate the batched notes
        batched_notes = cat_batched_notes(batched_notes, new_batched_notes)
        # Convert back to notes
        pitches, intervals = batched_notes_to_notes(batched_notes)
        # Overwrite the slice entry
        stacked_notes[slc] = pitches, intervals

    return stacked_notes


def filter_stacked_note_repeats(stacked_notes):
    """
    Remove any note duplicates within each slice, where a duplicate
    is defined as an entry with the same pitch and onset time.

    Parameters
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs

    Returns
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    """

    # Convert to batched notes
    stacked_notes = apply_func_stacked_representation(stacked_notes, notes_to_batched_notes)
    # Filter batched note repeats
    stacked_notes = apply_func_stacked_representation(stacked_notes, filter_batched_note_repeats)
    # Convert back to notes
    stacked_notes = apply_func_stacked_representation(stacked_notes, batched_notes_to_notes)

    return stacked_notes


def stacked_notes_to_frets(stacked_notes, tuning=None):
    """
    Convert stacked notes from MIDI to guitar fret numbers.

    Parameters
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches (MIDI), intervals)) pairs
    tuning : list of str
      Name of lowest note playable on each degree of freedom

    Returns
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches (fret), intervals)) pairs
    """

    # Make a copy of the stacked notes for conversion
    stacked_notes = deepcopy(stacked_notes)

    if tuning is None:
        # Default the tuning
        tuning = constants.DEFAULT_GUITAR_TUNING

    # Convert the tuning to midi pitches
    midi_tuning = librosa.note_to_midi(tuning)

    # Loop through the stack of notes
    for i, slc in enumerate(stacked_notes.keys()):
        # Get the notes from the slice
        pitches, intervals = stacked_notes[slc]
        # Convert the pitches to frets
        frets = np.round(pitches - midi_tuning[i]).astype(constants.UINT)
        # Add converted slice back to stack
        stacked_notes[slc] = frets, intervals

    return stacked_notes


def find_pitch_bounds_stacked_notes(stacked_notes):
    """
    Determine the minimum and maximum pitch for each slice of stacked notes.

    Parameters
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs

    Returns
    ----------
    min_pitches : ndarray (S)
      Minimum pitch across all notes per slice
      S - number of slices in stack
    max_pitches : ndarray (S)
      Maximum pitch across all notes per slice
      S - number of slices in stack
    """

    # Initialize lists to hold the pitch bounds
    min_pitches, max_pitches = list(), list()

    # Loop through the stack of notes
    for i, slc in enumerate(stacked_notes.keys()):
        # Get the pitches from the slice
        pitches, _ = stacked_notes[slc]

        # Add the minimum and maximum pitch for this slice
        min_pitches += [np.min(pitches) if len(pitches) > 0 else 0]
        max_pitches += [np.max(pitches) if len(pitches) > 0 else 0]

    # Convert to NumPy arrays and round to the nearest semitone
    min_pitches, max_pitches = np.round(np.array(min_pitches)), np.round(np.array(max_pitches))

    return min_pitches, max_pitches


##################################################
# TO PITCH LIST                                  #
##################################################


def stacked_pitch_list_to_pitch_list(stacked_pitch_list):
    """
    Convert a dictionary of stacked pitch lists into a single representation.

    Parameters
    ----------
    stacked_pitch_list : dict
      Dictionary containing (slice -> (times, pitch_list)) pairs

    Returns
    ----------
    times : ndarray (N)
      Time in seconds of beginning of each frame
      N - number of time samples (frames)
    pitch_list : list of ndarray (N x [...])
      Array of pitches corresponding to notes
      N - number of pitch observations (frames)
    """

    # Obtain the time-pitch list pairs from the dictionary values
    pitch_list_pairs = list(stacked_pitch_list.values())

    # Initialize an empty pitch list to start with
    times, pitch_list = np.array([]), []

    # Loop through each pitch list
    for slice_times, slice_pitch_list in pitch_list_pairs:
        # Concatenate the new pitch list with the current blend
        times, pitch_list = cat_pitch_list(times, pitch_list, slice_times, slice_pitch_list)

    # Sort the time-pitch array pairs by time
    times, pitch_list = sort_pitch_list(times, pitch_list)

    return times, pitch_list


def multi_pitch_to_pitch_list(multi_pitch, profile):
    """
    Convert a multi pitch array into a pitch list.
    Array of corresponding times does not change and is
    assumed to be managed outside of the function.

    Note: the result of this function should be
    invertible using pitch_list_to_multi_pitch()

    Parameters
    ----------
    multi_pitch : ndarray (F x T)
      Discrete pitch activation map
      F - number of discrete pitches
      T - number of frames
    profile : InstrumentProfile (instrument.py)
      Instrument profile detailing experimental setup

    Returns
    ----------
    pitch_list : list of ndarray (T x [...])
      Array of pitches corresponding to notes
      T - number of pitch observations (frames)
    """

    # Determine the number of frames in the multi pitch array
    num_frames = multi_pitch.shape[-1]

    # Initialize empty pitch arrays for each time entry
    pitch_list = [np.empty(0)] * num_frames

    # Determine which frames contain pitch activity
    non_silent_frames = np.where(np.sum(multi_pitch, axis=-2) > 0)[-1]

    # Loop through the frames containing pitch activity
    for i in list(non_silent_frames):
        # Determine the MIDI pitches active in the frame and add to the list
        pitch_list[i] = (profile.low + np.where(multi_pitch[..., i])[-1]).astype(constants.FLOAT)

    return pitch_list


def pitch_list_to_hz(pitch_list):
    """
    Convert pitch list from MIDI to Hertz.
    Array of corresponding times does not change and is
    assumed to be managed outside of the function.

    TODO - similar efficient implementations could be adopted elsewhere
           rather than using list comprehension for parsing pitch lists

    Parameters
    ----------
    pitch_list : list of ndarray (T x [...])
      Array of MIDI pitches corresponding to notes
      T - number of pitch observations (frames)

    Returns
    ----------
    pitch_list : list of ndarray (T x [...])
      Array of Hertz pitches corresponding to notes
      T - number of pitch observations (frames)
    """

    # Convert all pitches in the pitch list to Hertz
    all_pitches = librosa.midi_to_hz(np.concatenate(pitch_list))

    # Count the number of pitch observations at each frame
    frame_counts = get_active_pitch_count(pitch_list)
    # Determine which pitches belong to each frame
    pitch_idcs = np.append([0], np.cumsum(frame_counts))
    # Reconstruct the pitch list using the frame counts
    pitch_list = [all_pitches[pitch_idcs[k]: pitch_idcs[k + 1]]
                  for k in range(len(pitch_idcs) - 1)]

    return pitch_list


def pitch_list_to_midi(pitch_list):
    """
    Convert pitch list from Hertz to MIDI.
    Array of corresponding times does not change and is
    assumed to be managed outside of the function.

    Parameters
    ----------
    pitch_list : list of ndarray (T x [...])
      Array of Hertz pitches corresponding to notes
      T - number of pitch observations (frames)

    Returns
    ----------
    pitch_list : list of ndarray (T x [...])
      Array of MIDI pitches corresponding to notes
      T - number of pitch observations (frames)
    """

    # Convert to MIDI
    pitch_list = [librosa.hz_to_midi(pitch_list[i]) for i in range(len(pitch_list))]

    return pitch_list


def slice_pitch_list(times, pitch_list, start_time, stop_time):
    """
    Retain pitch observations within a time window.

    Parameters
    ----------
    times : ndarray (N)
      Time in seconds of beginning of each frame
      N - number of time samples (frames)
    pitch_list : list of ndarray (N x [...])
      Array of pitches active during each frame
      N - number of pitch observations (frames)
    start_time : float
      Earliest time for observations to keep
    stop_time : float
      Latest time for observations to keep

    Returns
    ----------
    times : ndarray (L)
      Time in seconds of beginning of each frame
      L - number of time samples (frames)
    pitch_list : list of ndarray (L x [...])
      Array of pitches active during each frame
      L - number of pitch observations (frames)
    """

    # Obtain a collection of valid indices for the slicing
    valid_idcs = np.logical_and((times >= start_time), (times <= stop_time))
    # Throw away observations occurring before the start time and after the stop time
    times, pitch_list = times[valid_idcs], [pitch_list[idx] for idx in np.where(valid_idcs)[0]]

    return times, pitch_list


def cat_pitch_list(times, pitch_list, new_times, new_pitch_list, decimals=6):
    """
    Concatenate two pitch lists, appending observations with new times and blending
    observations with preexisting times. This function assumes that all pitch lists
    have the same time grid. In order to avoid the influence of miniscule timing
    differences, timing comparisons are made with microsecond resolution.

    TODO - is there a less hacky way to circumvent float comparison issue?
           - see https://stackoverflow.com/questions/32513424/find-intersection-of-numpy-float-arrays
    TODO - could resample to make sure same time grid is used

    Parameters
    ----------
    times : ndarray (N)
      Time in seconds of beginning of each frame
      N - number of time samples (frames)
    pitch_list : list of ndarray (N x [...])
      Array of pitches active during each frame
      N - number of pitch observations (frames)
    new_times : ndarray (L)
      Time in seconds of beginning of each frame
      L - number of time samples (frames)
    new_pitch_list : list of ndarray (L x [...])
      Array of pitches active during each frame
      L - number of pitch observations (frames)
    decimals : int
      Decimal resolution for timing comparison

    Returns
    ----------
    times : ndarray (K)
      Time in seconds of beginning of each frame
      K - number of time samples (frames)
    pitch_list : list of ndarray (K x [...])
      Array of pitches active during each frame
      K - number of pitch observations (frames)
    """

    # Obtain microsecond resolution for both collections of times
    times_us, new_times_us = np.round(times * (10 ** decimals)), np.round(new_times * (10 ** decimals))

    # Indentify indices where new times overlap with original times
    overlapping_idcs_new = np.intersect1d(times_us, new_times_us, return_indices=True)[-1]

    # Count the number of new pitch observations at each frame
    new_frame_counts = get_active_pitch_count(new_pitch_list)

    # Determine which indices correspond to frames with observations
    non_empty_idcs_new = np.where(new_frame_counts != 0)[0]

    # Ignore indices where there are no new pitch observations
    overlapping_non_empty_idcs_new = np.intersect1d(overlapping_idcs_new, non_empty_idcs_new)

    # Determine which times overlap with the current pitch
    # list and are associated with new pitch observations
    overlapping_times = new_times_us[overlapping_non_empty_idcs_new]
    # Obtain the indices corresponding to the sorted times of the current pitch list
    sorter = times_us.argsort()
    # Find the mapping of the current times w.r.t. the new overlapping times
    corresponding_idcs = sorter[np.searchsorted(times_us, overlapping_times, sorter=sorter)]
    # Loop through all pairs of pitch list entries with matching times
    for (k, i) in zip(corresponding_idcs, overlapping_non_empty_idcs_new):
        # Combine the pitch observations into a single array
        pitch_list[k] = np.append(pitch_list[k], new_pitch_list[i])

    # Obtain the indices of new times which no not overlap with the original times
    non_overlapping_idcs_new = np.setdiff1d(np.arange(len(new_times)), overlapping_idcs_new)

    # Determine which (new) times do not exist in the original pitch list
    non_overlapping_times = new_times[non_overlapping_idcs_new]

    # Add the new entries to the pitch list
    times = np.append(times, non_overlapping_times)
    pitch_list = pitch_list + [new_pitch_list[i] for i in non_overlapping_idcs_new]

    # Make sure the blended pitch list is sorted
    times, pitch_list = sort_pitch_list(times, pitch_list)

    return times, pitch_list


def unroll_pitch_list(times, pitch_list):
    """
    Make a time-pitch pair for each active pitch in each frame.

    Parameters
    ----------
    times : ndarray (N)
      Time in seconds of beginning of each frame
      N - number of time samples (frames)
    pitch_list : list of ndarray (N x [...])
      Array of pitches active during each frame
      N - number of pitch observations (frames)

    Returns
    ----------
    times : ndarray (L)
      Time in seconds corresponding to pitch observations
      L - number of pitch observations
    pitches : ndarray (L)
      Array of pitch observations
    """

    # Repeat a frame time once for every active pitch in the frame and collapse into a single ndarray
    times = np.concatenate([[times[i]] * len(pitch_list[i]) for i in range(len(pitch_list))])

    # Collapse pitch list into a single ndarray
    pitches = np.concatenate(pitch_list, axis=-1)

    return times, pitches


def clean_pitch_list(pitch_list):
    """
    Remove null (zero-frequency) observations from the pitch list.

    Parameters
    ----------
    pitch_list : list of ndarray (N x [...])
      Array of pitches active during each frame
      N - number of pitch observations (frames)

    Returns
    ----------
    pitch_list : list of ndarray (N x [...])
      Original pitch list with no null observations
      N - number of pitch observations (frames)
    """

    # Loop through the pitch list and remove any zero-frequency entries
    pitch_list = [p[p != 0] for p in pitch_list]

    return pitch_list


def pack_pitch_list(times, pitch_list):
    """
    Package together the loose times and observations of a pitch list in a save-friendly format.

    Parameters
    ----------
    times : ndarray (N)
      Time in seconds of beginning of each frame
      N - number of time samples (frames)
    pitch_list : list of ndarray (N x [...])
      Array of pitches active during each frame
      N - number of pitch observations (frames)

    Returns
    ----------
    packed_pitch_list : ndarray (2 x N)
      Pitch list packaged in object ndarray
      N - number of loose pairs (frames)
    """

    # Concatenate the times and pitch list and wrap with object dtype
    # TODO - tools.pack_stacked_pitch_list(tools.pitch_list_to_stacked_pitch_list(times, pitch_list))?
    packed_pitch_list = np.array([times, np.array(pitch_list, dtype=object)])

    return packed_pitch_list


def unpack_pitch_list(packed_pitch_list):
    """
    Unpack from save-friendly format the loose times and observations of a pitch list.

    Parameters
    ----------
    packed_pitch_list : ndarray (2 x N)
      Pitch list packaged in object ndarray
      N - number of loose pairs (frames)

    Returns
    ----------
    times : ndarray (N)
      Time in seconds of beginning of each frame
      N - number of time samples (frames)
    pitch_list : list of ndarray (N x [...])
      Array of pitches active during each frame
      N - number of pitch observations (frames)
    """

    # Break apart along the concatenated dimension, convert
    # to 64-bit floats, and cast observations as a list
    # TODO - times, pitch_list = tools.stacked_pitch_list_to_pitch_list(tools.unpack_stacked_pitch_list(packed_pitch_list))?
    times = packed_pitch_list[0].astype(constants.FLOAT64)
    pitch_list = [p.astype(constants.FLOAT64) for p in packed_pitch_list[1]]

    return times, pitch_list


def get_active_pitch_count(pitch_list):
    """
    Count the number of active pitches in each frame of a pitch list.

    Parameters
    ----------
    pitch_list : list of ndarray (N x [...])
      Frame-level observations detailing active pitches
      N - number of frames

    Returns
    ----------
    active_pitch_count : ndarray
      Number of active pitches in each frame
    """

    # Make sure there are no null observations in the pitch list
    pitch_list = clean_pitch_list(pitch_list)
    # Determine the amount of non-zero frequencies in each frame
    active_pitch_count = np.array([len(p) for p in pitch_list])

    return active_pitch_count


def contains_empties_pitch_list(pitch_list):
    """
    Determine if a pitch list representation contains empty observations.

    Parameters
    ----------
    pitch_list : list of ndarray (N x [...])
      Frame-level observations detailing active pitches
      N - number of frames

    Returns
    ----------
    contains_empties : bool
      Whether there are any frames with no observations
    """

    # Check if at any time there are no observations
    contains_empties = np.sum(get_active_pitch_count(pitch_list) == 0) > 0

    return contains_empties


def detect_overlap_pitch_list(pitch_list):
    """
    Determine if a pitch list representation contains overlapping pitch contours.

    Parameters
    ----------
    pitch_list : list of ndarray (N x [...])
      Frame-level observations detailing active pitches
      N - number of frames

    Returns
    ----------
    overlap : bool
      Whether there are overlapping pitch contours
    """

    # Check if at any time there is more than one observation
    overlap = np.sum(get_active_pitch_count(pitch_list) > 1) > 0

    return overlap


def filter_pitch_list(pitch_list, profile, suppress_warnings=True):
    """
    Remove pitch observations (MIDI) outside the supported range of a profile.

    Parameters
    ----------
    pitch_list : list of ndarray (N x [...])
      Array of MIDI pitches active during each frame
      N - number of pitch observations (frames)
    profile : InstrumentProfile (instrument.py)
      Instrument profile detailing experimental setup
    suppress_warnings : bool
      Whether to ignore warning messages

    Returns
    ----------
    pitch_list : list of ndarray (N x [...])
      Original pitch list with no out-of-bounds observations
      N - number of pitch observations (frames)
    """

    # Only check and filter if the pitch list is not empty
    if np.sum(get_active_pitch_count(pitch_list)):
        # Flatten the pitch observations into an array
        flattened_observations = np.round(np.concatenate(pitch_list))

        if (np.min(flattened_observations) < profile.low or
            np.max(flattened_observations) > profile.high) and not suppress_warnings:
            # Print a warning message if pitch observations were ignored
            warnings.warn('Ignoring pitch observations exceeding ' +
                          'supported boundaries.', category=RuntimeWarning)

        # Loop through the pitch list and remove out-of-bounds frequency observations
        pitch_list = [p[np.logical_and((np.round(p) >= profile.low),
                                       (np.round(p) <= profile.high))] for p in pitch_list]

    return pitch_list


##################################################
# TO STACKED PITCH LIST                          #
##################################################


def pitch_list_to_stacked_pitch_list(times, pitch_list, i=0):
    """
    Convert a pitch list into a dictionary of stacked pitch lists.

    Parameters
    ----------
    times : ndarray (N)
      Time in seconds of beginning of each frame
      N - number of time samples (frames)
    pitch_list : list of ndarray (N x [...])
      Array of pitches corresponding to notes
      N - number of pitch observations (frames)
    i : int
      Slice key to use

    Returns
    ----------
    stacked_pitch_list : dict
      Dictionary containing (slice -> (times, pitch_list)) pairs
    """

    # Initialize a dictionary to hold the pitch_list
    stacked_pitch_list = dict()

    # Add the time-pitch array pairs to the stacked notes dictionary under the slice key
    stacked_pitch_list[i] = sort_pitch_list(times, pitch_list)

    return stacked_pitch_list


def stacked_multi_pitch_to_stacked_pitch_list(stacked_multi_pitch, times, profile):
    """
    Convert a stack of multi pitch arrays into a stack of pitch lists.

    Parameters
    ----------
    stacked_multi_pitch : ndarray (S x F x T)
      Array of multiple discrete pitch activation maps
      S - number of slices in stack
      F - number of discrete pitches
      T - number of frames
    times : ndarray (T)
      Time in seconds of beginning of each frame
      T - number of time samples (frames)
    profile : InstrumentProfile (instrument.py)
      Instrument profile detailing experimental setup

    Returns
    ----------
    stacked_pitch_list : dict
      Dictionary containing (slice -> (times, pitch_list)) pairs
    """

    # Determine the number of slices in the stacked multi pitch array
    stack_size = stacked_multi_pitch.shape[-3]

    # Initialize a dictionary to hold the pitch lists
    stacked_pitch_list = dict()

    # Loop through the slices of the stack
    for slc in range(stack_size):
        # Extract the multi pitch array pertaining to this slice
        slice_multi_pitch = stacked_multi_pitch[slc]

        # Convert the multi pitch array to a pitch list
        slice_pitch_list_ = multi_pitch_to_pitch_list(slice_multi_pitch, profile)

        # Add the pitch list to the stacked pitch list dictionary under the slice key
        stacked_pitch_list.update(pitch_list_to_stacked_pitch_list(times, slice_pitch_list_, slc))

    return stacked_pitch_list


def stacked_pitch_list_to_hz(stacked_pitch_list):
    """
    Convert stacked pitch list from MIDI to Hertz.

    Parameters
    ----------
    stacked_pitch_list : dict
      Dictionary containing (slice -> (times, pitch_list (MIDI))) pairs

    Returns
    ----------
    stacked_pitch_list : dict
      Dictionary containing (slice -> (times, pitch_list (Hertz)) pairs
    """

    # Make a copy of the stacked pitch lists for conversion
    stacked_pitch_list = deepcopy(stacked_pitch_list)

    # Loop through the stack of pitch lists
    for slc in stacked_pitch_list.keys():
        # Get the pitch list from the slice
        times, pitch_list = stacked_pitch_list[slc]
        # Convert the pitches to Hertz
        pitch_list = pitch_list_to_hz(pitch_list)
        # Add converted slice back to stack
        stacked_pitch_list[slc] = times, pitch_list

    return stacked_pitch_list


def stacked_pitch_list_to_midi(stacked_pitch_list):
    """
    Convert stacked pitch list from Hertz to MIDI.

    Parameters
    ----------
    stacked_pitch_list : dict
      Dictionary containing (slice -> (times, pitch_list (Hertz))) pairs

    Returns
    ----------
    stacked_pitch_list : dict
      Dictionary containing (slice -> (times, pitch_list (MIDI)) pairs
    """

    # Make a copy of the stacked pitch lists for conversion
    stacked_pitch_list = deepcopy(stacked_pitch_list)

    # Loop through the stack of pitch lists
    for slc in stacked_pitch_list.keys():
        # Get the pitches from the slice
        times, pitch_list = stacked_pitch_list[slc]
        # Convert the pitches to MIDI
        pitch_list = pitch_list_to_midi(pitch_list)
        # Add converted slice back to stack
        stacked_pitch_list[slc] = times, pitch_list

    return stacked_pitch_list


def slice_stacked_pitch_list(stacked_pitch_list, start_time, stop_time):
    """
    Retain pitch observations within a time window.

    Parameters
    ----------
    stacked_pitch_list : dict
      Dictionary containing (slice -> (times, pitch_list)) pairs
    start_time : float
      Earliest time for observations to keep
    stop_time : float
      Latest time for observations to keep

    Returns
    ----------
    stacked_pitch_list : dict
      Dictionary containing (slice -> (times, pitch_list)) pairs
    """

    # Make a copy of the stacked pitch lists for slicing
    stacked_pitch_list = deepcopy(stacked_pitch_list)

    # Loop through the stack of pitch lists
    for slc in stacked_pitch_list.keys():
        # Get the times and pitches from the slice
        times, pitch_list = stacked_pitch_list[slc]
        # Slice the pitch list for this slice
        stacked_pitch_list[slc] = slice_pitch_list(times, pitch_list, start_time, stop_time)

    return stacked_pitch_list


def cat_stacked_pitch_list(stacked_pitch_list, new_stacked_pitch_list):
    """
    Concatenate two stacked pitch lists, appending observations with
    new times and blending observations with preexisting times. This
    function assumes that all pitch lists have the same time grid.

    Parameters
    ----------
    stacked_pitch_list : dict
      Dictionary containing (slice -> (times, pitch_list)) pairs
    new_stacked_pitch_list : dict
      Same as stacked_pitch_list
      Note: must also have same number of slices

    Returns
    ----------
    stacked_pitch_list : dict
      Dictionary containing (slice -> (times, pitch_list)) pairs
    """

    # Make a copy of the stacked pitch lists for concatenation
    stacked_pitch_list = deepcopy(stacked_pitch_list)

    # Loop through the stack of pitch lists
    for slc in stacked_pitch_list.keys():
        # Group the two pitch_lists at the slice and add back to stack
        stacked_pitch_list[slc] = cat_pitch_list(*(stacked_pitch_list[slc] + new_stacked_pitch_list[slc]))

    return stacked_pitch_list


##################################################
# TO MULTI PITCH                                 #
##################################################


def notes_to_multi_pitch(pitches, intervals, times, profile, include_offsets=True):
    """
    Convert loose MIDI note groups into a multi pitch array.

    Parameters
    ----------
    pitches : ndarray (N)
      Array of pitches corresponding to notes in MIDI format
      N - number of notes
    intervals : ndarray (N x 2)
      Array of onset-offset time pairs corresponding to notes
      N - number of notes
    times : ndarray (N)
      Time in seconds of beginning of each frame
      N - number of time samples (frames)
    profile : InstrumentProfile (instrument.py)
      Instrument profile detailing experimental setup
    include_offsets : bool
      Whether to include an activation at the very last frame of a note

    Returns
    ----------
    multi_pitch : ndarray (F x T)
      Discrete pitch activation map
      F - number of discrete pitches
      T - number of frames
    """

    # Determine the dimensionality of the multi pitch array
    num_pitches = profile.get_range_len()
    num_frames = len(times)

    # Initialize an empty multi pitch array
    multi_pitch = np.zeros((num_pitches, num_frames))

    # Estimate the duration of the track (for bounding note offsets)
    _times = np.append(times, times[-1] + estimate_hop_length(times))

    # Remove notes with out-of-bounds intervals or nominal pitch
    pitches, intervals = filter_notes(pitches, intervals, profile,
                                      min_time=np.min(_times),
                                      max_time=np.max(_times))

    # Count the total number of notes
    num_notes = len(pitches)

    # Round to nearest semitone and subtract the lowest
    # note of the instrument to obtain relative pitches
    pitches = np.round(pitches - profile.low).astype(constants.INT)

    # Duplicate the array of times for each note and stack along a new axis
    # TODO - should onset/offset determination be under the for loop?
    #        current methodology may not scale well w.r.t. memory and
    #        we are already doing a for-loop
    times_broadcast = np.concatenate([[_times]] * max(1, num_notes), axis=0)

    # Determine the frame where each note begins and ends, defined
    # for both onset and offset as the last frame beginning before
    # (or at the same time as) that of the respective event
    # TODO - should offsets comparison be < instead of <=?
    onsets = np.argmin((times_broadcast <= intervals[..., :1]), axis=1) - 1
    offsets = np.argmin((times_broadcast <= intervals[..., 1:]), axis=1) - 1

    # Clip all onsets/offsets at first/last frame - these will end up
    # at -1 from previous operation if they occurred beyond boundaries
    onsets[onsets == -1], offsets[offsets == -1] = 0, num_frames - 1

    # Loop through each note
    for i in range(num_notes):
        # Populate the multi pitch array with activations for the note
        multi_pitch[pitches[i], onsets[i] : offsets[i] + int(include_offsets)] = 1

    return multi_pitch


def pitch_list_to_multi_pitch(pitch_list, profile):
    """
    Convert a MIDI pitch list into a multi pitch array.

    Parameters
    ----------
    pitch_list : list of ndarray (N x [...])
      Array of MIDI pitches corresponding to notes
      N - number of pitch observations (frames)
    profile : InstrumentProfile (instrument.py)
      Instrument profile detailing experimental setup

    Note: the result of this function should be
    invertible (within a 0.5 semitone tolerance)
    using multi_pitch_to_pitch_list(), assuming no
    out-of-bounds pitches exist in the pitch list

    Returns
    ----------
    multi_pitch : ndarray (F x T)
      Discrete pitch activation map
      F - number of discrete pitches
      T - number of frames
    """

    # Throw away out-of-bounds pitche observations
    pitch_list = filter_pitch_list(pitch_list, profile)

    # Determine the dimensionality of the multi pitch array
    num_pitches = profile.get_range_len()
    num_frames = len(pitch_list)

    # Initialize an empty multi pitch array
    multi_pitch = np.zeros((num_pitches, num_frames))

    # Loop through each frame
    for i in range(len(pitch_list)):
        # Calculate the pitch semitone difference from the lowest note
        difference = pitch_list[i] - profile.low
        # Convert the pitches to number of semitones from lowest note
        pitch_idcs = np.round(difference).astype(constants.UINT)
        # Populate the multi pitch array with activations
        multi_pitch[pitch_idcs, i] = 1

    return multi_pitch


def stacked_multi_pitch_to_multi_pitch(stacked_multi_pitch):
    """
    Collapse stacked multi pitch arrays into a single representation.

    Parameters
    ----------
    stacked_multi_pitch : ndarray or tensor (..., S x F x T)
      Array of multiple discrete pitch activation maps
      S - number of slices in stack
      F - number of discrete pitches
      T - number of frames

    Returns
    ----------
    multi_pitch : ndarray or tensor (..., F x T)
      Discrete pitch activation map
      F - number of discrete pitches
      T - number of frames
    """

    # Collapse the stacked arrays into one using the max operation
    if isinstance(stacked_multi_pitch, torch.Tensor):
        # PyTorch Tensor
        multi_pitch = torch.max(stacked_multi_pitch, dim=-3)[0]
    else:
        # NumPy Ndarray
        multi_pitch = np.max(stacked_multi_pitch, axis=-3)

    return multi_pitch


def logistic_to_stacked_multi_pitch(logistic, profile, silence=True):
    """
    View logistic activations as a stacked multi pitch array.

    Parameters
    ----------
    logistic : ndarray or tensor (..., N x T)
      Array of distinct activations (e.g. string/fret combinations)
      N - number of individual activations
      T - number of frames
    profile : TablatureProfile (instrument.py)
      Tablature instrument profile detailing experimental setup
    silence : bool
      Whether there is an activation for silence

    Returns
    ----------
    stacked_multi_pitch : ndarray or tensor (..., S x F x T)
      Array of multiple discrete pitch activation maps
      S - number of slices in stack
      F - number of discrete pitches
      T - number of frames
    """

    # Obtain the tuning (lowest note for each degree of freedom)
    tuning = profile.get_midi_tuning()

    # Determine the appropriate dimensionality of the stacked multi pitch array
    dims = tuple(logistic.shape[:-2] +
                 (len(tuning), profile.get_range_len(), logistic.shape[-1]))

    if isinstance(logistic, np.ndarray):
        # Initialize an array of zeros
        stacked_multi_pitch = np.zeros(dims)
    else:
        # Initialize a tensor of zeros
        stacked_multi_pitch = torch.zeros(dims, device=logistic.device)

    # Loop through the degrees of freedom
    for dof in range(dims[-3]):
        # Determine which activations correspond to this degree of freedom
        start_idx = dof * (profile.num_pitches + int(silence))
        stop_idx = (dof + 1) * (profile.num_pitches + int(silence))

        # Obtain the logistic activations for the degree of freedom
        activations = logistic[..., start_idx + int(silence) : stop_idx, :]

        # Lower and upper pitch boundary for this degree of freedom
        lower_bound = tuning[dof] - profile.low
        upper_bound = lower_bound + profile.num_pitches

        # Insert the activations for this degree of freedom with the appropriate offset
        stacked_multi_pitch[..., dof, lower_bound : upper_bound, :] = activations

    return stacked_multi_pitch


##################################################
# TO STACKED MULTI PITCH                         #
##################################################


def stacked_notes_to_stacked_multi_pitch(stacked_notes, times, profile, include_offsets=True):
    """
    Convert a dictionary of MIDI note groups into a stack of multi pitch arrays.

    Parameters
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    times : ndarray (N)
      Time in seconds of beginning of each frame
      N - number of time samples (frames)
    profile : InstrumentProfile (instrument.py)
      Instrument profile detailing experimental setup
    include_offsets : bool
      Whether to include an activation at the very last frame of a note

    Returns
    ----------
    stacked_multi_pitch : ndarray (S x F x T)
      Array of multiple discrete pitch activation maps
      S - number of slices in stack
      F - number of discrete pitches
      T - number of frames
    """

    # Initialize an empty list to hold the multi pitch arrays
    stacked_multi_pitch = list()

    # Loop through the slices of notes
    for slc in stacked_notes.keys():
        # Get the pitches and intervals from the slice
        pitches, intervals = stacked_notes[slc]
        # Convert to multi pitch and add to the list
        slice_multi_pitch = notes_to_multi_pitch(pitches, intervals, times, profile, include_offsets)
        stacked_multi_pitch.append(multi_pitch_to_stacked_multi_pitch(slice_multi_pitch))

    # Collapse the list into an array
    stacked_multi_pitch = np.concatenate(stacked_multi_pitch)

    return stacked_multi_pitch


def stacked_pitch_list_to_stacked_multi_pitch(stacked_pitch_list, profile):
    """
    Convert a stacked MIDI pitch list into a stack of multi pitch arrays.
    This function assumes that all pitch lists are relative to the same
    timing grid.

    Parameters
    ----------
    stacked_pitch_list : dict
      Dictionary containing (slice -> (times, pitch_list)) pairs
    profile : InstrumentProfile (instrument.py)
      Instrument profile detailing experimental setup

    Returns
    ----------
    stacked_multi_pitch : ndarray (S x F x T)
      Array of multiple discrete pitch activation maps
      S - number of slices in stack
      F - number of discrete pitches
      T - number of frames
    """

    # Initialize an empty list to hold the multi pitch arrays
    stacked_multi_pitch = list()

    # Loop through the slices of notes
    for slc in stacked_pitch_list.keys():
        # Get the pitch observations from the slice
        _, pitch_list = stacked_pitch_list[slc]
        # Convert the pitch observations to multi pitch activations
        multi_pitch = pitch_list_to_multi_pitch(pitch_list, profile)
        # Add the multi pitch activations to the list
        stacked_multi_pitch.append(multi_pitch_to_stacked_multi_pitch(multi_pitch))

    # Stack all of the multi pitch arrays along the first dimension
    stacked_multi_pitch = np.concatenate(stacked_multi_pitch)

    return stacked_multi_pitch


def multi_pitch_to_stacked_multi_pitch(multi_pitch):
    """
    Convert a multi pitch array into a stacked representation.

    Parameters
    ----------
    multi_pitch : ndarray (F x T)
      Discrete pitch activation map
      F - number of discrete pitches
      T - number of frames

    Returns
    ----------
    stacked_multi_pitch : ndarray (S x F x T)
      Array of multiple discrete pitch activation maps
      S - number of slices in stack
      F - number of discrete pitches
      T - number of frames
    """

    # Add an extra dimension for slice
    stacked_multi_pitch = np.expand_dims(multi_pitch, axis=-3)

    return stacked_multi_pitch


def tablature_to_stacked_multi_pitch(tablature, profile):
    """
    Convert a tablature representation into a stacked multi pitch array.
    Array of corresponding times does not change and is
    assumed to be managed outside of the function.

    Parameters
    ----------
    tablature : ndarray or tensor (..., S x T) (must consist of integers)
      Array of class membership for multiple degrees of freedom (e.g. strings)
      S - number of strings or degrees of freedom
      T - number of frames
    profile : TablatureProfile (instrument.py)
      Tablature instrument profile detailing experimental setup

    Returns
    ----------
    stacked_multi_pitch : ndarray or tensor (..., S x F x T)
      Array of multiple discrete pitch activation maps
      S - number of slices in stack
      F - number of discrete pitches
      T - number of frames
    """

    # Determine the number of degrees of freedom and frames
    num_dofs, num_frames = tablature.shape[-2:]

    # Determine the total number of pitches to be incldued
    num_pitches = profile.get_range_len()

    # Initialize and empty stacked multi pitch array
    stacked_multi_pitch = np.zeros(tablature.shape[:-2] + (num_dofs, num_pitches, num_frames))

    # Obtain the tuning for the tablature (lowest note for each degree of freedom)
    tuning = profile.get_midi_tuning()

    # Determine the place in the stacked multi pitch array where each degree of freedom begins
    dof_start = np.expand_dims(tuning - profile.low, -1)

    if isinstance(tablature, torch.Tensor):
        # Convert these to tensor
        dof_start = torch.Tensor(dof_start).to(tablature.device)

    # Determine which frames, by degree of freedom, contain pitch activity
    non_silent_frames = tablature >= 0

    # Determine the active pitches, relative to the start of the stacked multi pitch array
    pitch_idcs = (tablature + dof_start)[non_silent_frames]

    # Obtain the non-silent indices across each dimension
    non_silent_idcs = non_silent_frames.nonzero()

    if isinstance(tablature, torch.Tensor):
        # Make sure pitch indices are integers
        pitch_idcs = pitch_idcs.long()
        # Extra step for PyTorch Tensors (tuple with indices for each dimension)
        non_silent_idcs = tuple(non_silent_idcs.transpose(-2, -1))
        # Convert to Tensor and add to the appropriate device
        stacked_multi_pitch = torch.from_numpy(stacked_multi_pitch).to(tablature.device)
        # Make sure the tensor has the same type as the input tablature
        stacked_multi_pitch = stacked_multi_pitch.to(tablature.dtype)
    else:
        # Make sure pitch indices are integers
        pitch_idcs = pitch_idcs.astype(constants.INT64)

    # Split the non-silent indices by frame vs. everything else
    other_idcs, frame_idcs = non_silent_idcs[:-1], non_silent_idcs[-1]

    # Populate the stacked multi pitch array
    stacked_multi_pitch[other_idcs + (pitch_idcs, frame_idcs)] = 1

    return stacked_multi_pitch


##################################################
# TO TABLATURE                                   #
##################################################


def stacked_pitch_list_to_tablature(stacked_pitch_list, profile):
    """
    Convert a stacked MIDI pitch list into a single class representation.

    Parameters
    ----------
    stacked_pitch_list : dict
      Dictionary containing (slice -> (times, pitch_list)) pairs
    profile : TablatureProfile (instrument.py)
      Tablature instrument profile detailing experimental setup

    Returns
    ----------
    tablature : ndarray (S x T)
      Array of class membership for multiple degrees of freedom (e.g. strings)
      S - number of strings or degrees of freedom
      T - number of frames
    """

    # Convert the stacked pitch list into a stacked multi pitch representation
    stacked_multi_pitch = stacked_pitch_list_to_stacked_multi_pitch(stacked_pitch_list, profile)

    # Convert the stacked multi pitch array into tablature
    tablature = stacked_multi_pitch_to_tablature(stacked_multi_pitch, profile)

    return tablature


def stacked_multi_pitch_to_tablature(stacked_multi_pitch, profile):
    """
    Collapse stacked multi pitch arrays into a single class representation.

    Parameters
    ----------
    stacked_multi_pitch : ndarray (S x F x T)
      Array of multiple discrete pitch activation maps
      S - number of slices in stack
      F - number of discrete pitches
      T - number of frames
    profile : TablatureProfile (instrument.py)
      Tablature instrument profile detailing experimental setup

    Returns
    ----------
    tablature : ndarray (S x T)
      Array of class membership for multiple degrees of freedom (e.g. strings)
      S - number of strings or degrees of freedom
      T - number of frames
    """

    # Obtain the tuning for the tablature (lowest note for each degree of freedom)
    tuning = profile.get_midi_tuning()

    # Initialize an empty list to hold the tablature
    tablature = list()

    # Loop through the multi pitch arrays
    for dof in range(len(stacked_multi_pitch)):
        # Obtain the multi pitch array for the degree of freedom
        multi_pitch = stacked_multi_pitch[dof]

        # Lower and upper pitch boundary for this degree of freedom
        lower_bound = tuning[dof] - profile.low
        upper_bound = lower_bound + profile.num_pitches

        # Bound the multi pitch array by the support of the degree of freedom
        multi_pitch = multi_pitch[lower_bound : upper_bound]

        # Determine which frames have no note activations
        silent_frames = np.sum(multi_pitch, axis=0) == 0

        # Determine which class has the highest activation across each frame
        highest_class = np.argmax(multi_pitch, axis=0)

        # Overwrite the highest class for the silent frames
        highest_class[silent_frames] = -1

        # Add the class membership to the tablature
        tablature += [np.expand_dims(highest_class, axis=0)]

    # Collapse the list to get the final tablature
    tablature = np.concatenate(tablature)

    return tablature


def logistic_to_tablature(logistic, profile, silence, silence_thr=0.05):
    """
    View logistic activations as tablature class membership.

    Parameters
    ----------
    logistic : tensor (..., N x T)
      Array of distinct activations (e.g. string/fret combinations)
      N - number of individual activations
      T - number of frames
    profile : TablatureProfile (instrument.py)
      Tablature instrument profile detailing experimental setup
    silence : bool
      Whether there is an activation for silence
    silence_thr : float
      Threshold for maximum activation under which silence will be selected

    Returns
    ----------
    tablature : tensor (S x T)
      Array of class membership for multiple degrees of freedom (e.g. strings)
      S - number of strings or degrees of freedom
      T - number of frames
    """

    # Obtain the tuning (lowest note for each degree of freedom)
    tuning = profile.get_midi_tuning()

    # Initialize an empty list to hold the tablature
    tablature = list()

    # Loop through the multi pitch arrays
    for dof in range(len(tuning)):
        # Determine which activations correspond to this degree of freedom
        start_idx = dof * (profile.num_pitches + int(silence))
        stop_idx = (dof + 1) * (profile.num_pitches + int(silence))

        # Obtain the logistic activations for the degree of freedom
        activations = logistic[..., start_idx : stop_idx, :]

        # Determine which class has the highest activation across each frame
        if isinstance(logistic, np.ndarray):
            max_activations, highest_class = np.max(activations, axis=-2), np.argmax(activations, axis=-2)
        else:
            max_activations, highest_class = torch.max(activations, axis=-2)

        if silence:
            highest_class -= 1
        else:
            # Determine which frames correspond to silence
            silent_frames = max_activations <= silence_thr
            # Overwrite the highest class for the silent frames
            highest_class[silent_frames] = -1

        # Add the class membership to the tablature
        if isinstance(logistic, np.ndarray):
            tablature += [np.expand_dims(highest_class, axis=-2)]
        else:
            tablature += [highest_class.unsqueeze(-2)]

    # Collapse the list to get the final tablature
    if isinstance(logistic, np.ndarray):
        tablature = np.concatenate(tablature, axis=-2)
    else:
        tablature = torch.cat(tablature, dim=-2)

    return tablature


##################################################
# TO LOGISTIC                                    #
##################################################


def stacked_multi_pitch_to_logistic(stacked_multi_pitch, profile, silence=False):
    """
    View stacked multi pitch arrays as a set of individual activations.

    Parameters
    ----------
    stacked_multi_pitch : ndarray (S x F x T)
      Array of multiple discrete pitch activation maps
      S - number of slices in stack
      F - number of discrete pitches
      T - number of frames
    profile : TablatureProfile (instrument.py)
      Tablature instrument profile detailing experimental setup
    silence : bool
      Whether to explicitly include an activation for silence

    Returns
    ----------
    logistic : ndarray (N x T)
      Array of distinct activations (e.g. string/fret combinations)
      N - number of individual activations
      T - number of frames
    """

    # Obtain the tuning (lowest note for each degree of freedom)
    tuning = profile.get_midi_tuning()

    # Initialize an empty list to hold the logistic activations
    logistic = list()

    # Loop through the multi pitch arrays
    for dof in range(stacked_multi_pitch.shape[-3]):
        # Obtain the multi pitch array for the degree of freedom
        multi_pitch = stacked_multi_pitch[..., dof, :, :]

        # Lower and upper pitch boundary for this degree of freedom
        lower_bound = tuning[dof] - profile.low
        upper_bound = lower_bound + profile.num_pitches

        # Bound the multi pitch array by the support of the degree of freedom
        multi_pitch = multi_pitch[..., lower_bound : upper_bound, :]

        if silence:
            if isinstance(multi_pitch, np.ndarray):
                # Construct an array with activations for silence
                silence_activations = np.sum(multi_pitch, axis=-2, keepdims=True) == 0
                # Append the silence activations at the front of the multi pitch array
                multi_pitch = np.append(silence_activations, multi_pitch, axis=-2)
            else:
                # Construct a tensor with activations for silence
                silence_activations = torch.sum(multi_pitch, dim=-2, keepdims=True) == 0
                # Append the silence activations at the front of the multi pitch array
                multi_pitch = torch.cat((silence_activations.to(multi_pitch.device), multi_pitch), dim=-2)

        # Add the multi pitch data to the logistic activations
        logistic += [multi_pitch]

    # Collapse the list to get the final logistic activations
    if isinstance(stacked_multi_pitch, np.ndarray):
        logistic = np.concatenate(logistic, axis=-2)
    else:
        logistic = torch.cat(logistic, dim=-2)

    return logistic


def tablature_to_logistic(tablature, profile, silence=False):
    """
    Helper function to convert tablature to unique string/fret combinations.

    Parameters
    ----------
    tablature : tensor (S x T)
      Array of class membership for multiple degrees of freedom (e.g. strings)
      S - number of strings or degrees of freedom
      T - number of frames
    profile : TablatureProfile (instrument.py)
      Tablature instrument profile detailing experimental setup
    silence : bool
      Whether to explicitly include an activation for silence

    Returns
    ----------
    logistic_activations : ndarray (N x T)
      Array of tablature activations (e.g. string/fret combinations)
      N - number of unique string/fret activations
      T - number of frames
    """

    # Convert the tablature data to a stacked multi pitch array
    stacked_multi_pitch = tablature_to_stacked_multi_pitch(tablature, profile)

    # Convert the stacked multi pitch array to logistic (unique string/fret) activations
    logistic_activations = stacked_multi_pitch_to_logistic(stacked_multi_pitch, profile, silence)

    return logistic_activations

##################################################
# TO ONSETS                                      #
##################################################


def notes_to_onsets(pitches, intervals, times, profile, ambiguity=None):
    """
    Obtain the onsets of loose MIDI note groups in multi pitch format.

    Parameters
    ----------
    pitches : ndarray (N)
      Array of pitches corresponding to notes in MIDI format
      N - number of notes
    intervals : ndarray (N x 2)
      Array of onset-offset time pairs corresponding to notes
      N - number of notes
    times : ndarray (N)
      Time in seconds of beginning of each frame
      N - number of time samples (frames)
    profile : InstrumentProfile (instrument.py)
      Instrument profile detailing experimental setup
    ambiguity : float or None (optional)
      Amount of time each onset label should span

    Returns
    ----------
    onsets : ndarray (F x T)
      Discrete onset activation map
      F - number of discrete pitches
      T - number of frames
    """

    # Determine the absolute time of each onset and offset
    onset_times = np.copy(intervals[..., :1])
    offset_times = np.copy(intervals[..., 1:])

    if ambiguity is not None:
        # Obtain the duration of each note
        durations = offset_times - onset_times
        # Truncate the note lengths
        durations = np.minimum(durations, ambiguity)
        # Set the offset times to match the truncated length
        offset_times = onset_times + durations
    else:
        # Only mark the frame where the onset happens as an activation
        offset_times = np.copy(onset_times)

    # Construct the intervals of the truncated note following the onset
    truncated_note_intervals = np.concatenate((onset_times, offset_times), axis=-1)

    # Obtain the offsets using the note to multi pitch conversion
    onsets = notes_to_multi_pitch(pitches, truncated_note_intervals, times, profile)

    return onsets


def multi_pitch_to_onsets(multi_pitch):
    """
    Obtain a representation detailing where discrete pitches become active.

    Parameters
    ----------
    multi_pitch : ndarray (F x T)
      Discrete pitch activation map
      F - number of discrete pitches
      T - number of frames

    Returns
    ----------
    onsets : ndarray (F x T)
      Discrete onset activation map
      F - number of discrete pitches
      T - number of frames
    """

    # Any pitches active in the first frame are considered onsets
    first_frame = multi_pitch[..., :1]

    # Subtract adjacent frames to determine where activity begins
    adjacent_diff = multi_pitch[..., 1:] - multi_pitch[..., :-1]

    # Combine the previous observations into a single representation
    onsets = np.concatenate([first_frame, adjacent_diff], axis=-1)

    # Consider anything above zero an onset
    onsets[onsets <= 0] = 0

    return onsets


##################################################
# TO STACKED ONSETS                              #
##################################################


def stacked_notes_to_stacked_onsets(stacked_notes, times, profile, ambiguity=None):
    """
    Obtain the onsets of stacked loose MIDI note groups in stacked multi pitch format.

    Parameters
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches (MIDI), intervals)) pairs
    times : ndarray (N)
      Time in seconds of beginning of each frame
      N - number of time samples (frames)
    profile : InstrumentProfile (instrument.py)
      Instrument profile detailing experimental setup
    ambiguity : float or None (optional)
      Amount of time each onset label should span

    Returns
    ----------
    stacked_onsets : ndarray (S x F x T)
      Array of multiple discrete onset activation maps
      S - number of slices in stack
      F - number of discrete pitches
      T - number of frames
    """

    # Initialize an empty list to hold the onset arrays
    stacked_onsets = list()

    # Loop through the slices of notes
    for slc in stacked_notes.keys():
        # Get the pitches and intervals from the slice
        pitches, intervals = stacked_notes[slc]
        # Convert to onsets and add to the list
        slice_onsets = notes_to_onsets(pitches, intervals, times, profile, ambiguity)
        stacked_onsets.append(multi_pitch_to_stacked_multi_pitch(slice_onsets))

    # Collapse the list into an array
    stacked_onsets = np.concatenate(stacked_onsets)

    return stacked_onsets


def stacked_multi_pitch_to_stacked_onsets(stacked_multi_pitch):
    """
    Obtain a stacked representation detailing where discrete pitches become active.

    Parameters
    ----------
    stacked_multi_pitch : ndarray (S x F x T)
      Array of multiple discrete pitch activation maps
      S - number of slices in stack
      F - number of discrete pitches
      T - number of frames

    Returns
    ----------
    stacked_onsets : ndarray (S x F x T)
      Array of multiple discrete onset activation maps
      S - number of slices in stack
      F - number of discrete pitches
      T - number of frames
    """

    # Determine the number of slices in the stacked multi pitch array
    stack_size = stacked_multi_pitch.shape[-3]

    # Initialize an empty list to hold the onset arrays
    stacked_onsets = list()

    # Loop through the slices of the stack
    for slc in range(stack_size):
        # Extract the multi pitch array pertaining to this slice
        slice_multi_pitch = stacked_multi_pitch[slc]
        # Convert to onsets and add to the list
        slice_onsets = multi_pitch_to_onsets(slice_multi_pitch)
        stacked_onsets.append(multi_pitch_to_stacked_multi_pitch(slice_onsets))

    # Collapse the list into an array
    stacked_onsets = np.concatenate(stacked_onsets)

    return stacked_onsets


##################################################
# TO OFFSETS                                     #
##################################################


def notes_to_offsets(pitches, intervals, times, profile, ambiguity=None):
    """
    Obtain the offsets of loose MIDI note groups in multi pitch format.

    Parameters
    ----------
    pitches : ndarray (N)
      Array of pitches corresponding to notes in MIDI format
      N - number of notes
    intervals : ndarray (N x 2)
      Array of onset-offset time pairs corresponding to notes
      N - number of notes
    times : ndarray (N)
      Time in seconds of beginning of each frame
      N - number of time samples (frames)
    profile : InstrumentProfile (instrument.py)
      Instrument profile detailing experimental setup
    ambiguity : float or None (optional)
      Amount of time each offset label should span

    Returns
    ----------
    offsets : ndarray (F x T)
      Discrete offset activation map
      F - number of discrete pitches
      T - number of frames
    """

    # Determine the absolute time of the offset
    offset_times = np.copy(intervals[..., 1:])

    # Make the duration zero
    onset_times = np.copy(offset_times)

    if ambiguity is not None:
        # Add the ambiguity to the "note" duration
        offset_times += ambiguity

    # Construct the intervals of the "note" following the offset
    post_note_intervals = np.concatenate((onset_times, offset_times), axis=-1)

    # Obtain the offsets using the note to multi pitch conversion
    offsets = notes_to_multi_pitch(pitches, post_note_intervals, times, profile)

    return offsets


def multi_pitch_to_offsets(multi_pitch):
    """
    Obtain a representation detailing where discrete pitch activity ceases.

    Parameters
    ----------
    multi_pitch : ndarray (F x T)
      Discrete pitch activation map
      F - number of discrete pitches
      T - number of frames

    Returns
    ----------
    offsets : ndarray (F x T)
      Discrete offset activation map
      F - number of discrete pitches
      T - number of frames
    """

    # Any pitches active in the last frame are considered offsets
    last_frame = multi_pitch[..., -1:]

    # Subtract adjacent frames to determine where activity ceases
    adjacent_diff = multi_pitch[..., 1:] - multi_pitch[..., :-1]

    # Flip the differentials so negative become positive and vise-versa
    adjacent_diff = -1 * adjacent_diff

    # Combine the previous observations into a single representation
    offsets = np.concatenate([adjacent_diff, last_frame], axis=-1)

    # Consider anything below zero an offset
    offsets[offsets <= 0] = 0

    return offsets


##################################################
# TO STACKED OFFSETS                             #
##################################################


def stacked_notes_to_stacked_offsets(stacked_notes, times, profile, ambiguity=None):
    """
    Obtain the offsets of stacked loose MIDI note groups in stacked multi pitch format.

    Parameters
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches (MIDI), intervals)) pairs
    times : ndarray (N)
      Time in seconds of beginning of each frame
      N - number of time samples (frames)
    profile : InstrumentProfile (instrument.py)
      Instrument profile detailing experimental setup
    ambiguity : float or None (optional)
      Amount of time each onset label should span

    Returns
    ----------
    stacked_offsets : ndarray (S x F x T)
      Array of multiple discrete offset activation maps
      S - number of slices in stack
      F - number of discrete pitches
      T - number of frames
    """

    # Initialize an empty list to hold the offset arrays
    stacked_offsets = list()

    # Loop through the slices of notes
    for slc in stacked_notes.keys():
        # Get the pitches and intervals from the slice
        pitches, intervals = stacked_notes[slc]
        # Convert to offsets and add to the list
        slice_offsets = notes_to_offsets(pitches, intervals, times, profile, ambiguity)
        stacked_offsets.append(multi_pitch_to_stacked_multi_pitch(slice_offsets))

    # Collapse the list into an array
    stacked_offsets = np.concatenate(stacked_offsets)

    return stacked_offsets


def stacked_multi_pitch_to_stacked_offsets(stacked_multi_pitch):
    """
    Obtain a stacked representation detailing where discrete pitch activity ceases.

    Parameters
    ----------
    stacked_multi_pitch : ndarray (S x F x T)
      Array of multiple discrete pitch activation maps
      S - number of slices in stack
      F - number of discrete pitches
      T - number of frames

    Returns
    ----------
    stacked_offsets : ndarray (S x F x T)
      Array of multiple discrete offset activation maps
      S - number of slices in stack
      F - number of discrete pitches
      T - number of frames
    """

    # Determine the number of slices in the stacked multi pitch array
    stack_size = stacked_multi_pitch.shape[-3]

    # Initialize an empty list to hold the offset arrays
    stacked_offsets = list()

    # Loop through the slices of the stack
    for slc in range(stack_size):
        # Extract the multi pitch array pertaining to this slice
        slice_multi_pitch = stacked_multi_pitch[slc]
        # Convert to offsets and add to the list
        slice_offsets = multi_pitch_to_offsets(slice_multi_pitch)
        stacked_offsets.append(multi_pitch_to_stacked_multi_pitch(slice_offsets))

    # Collapse the list into an array
    stacked_offsets = np.concatenate(stacked_offsets)

    return stacked_offsets


##################################################
# SORTING                                        #
##################################################


def sort_batched_notes(batched_notes, by=0):
    """
    Sort an array of batch-friendly notes by the specified attribute.

    Parameters
    ----------
    batched_notes : ndarray (N x 3)
      Array of note pitches and intervals by row
      N - number of notes
    by : int
      Index to sort notes by
      0 - onset | 1 - offset | 2 - pitch

    Returns
    ----------
    batched_notes : ndarray (N x 3)
      Array of note pitches and intervals by row, sorted by selected attribute
      N - number of notes
    """

    # Obtain the indices of the array to sort by the chosen attribute
    sorted_idcs = np.argsort(batched_notes[..., by])
    # Re-order the array according to the sorting indices
    batched_notes = batched_notes[sorted_idcs]

    return batched_notes


def sort_notes(pitches, intervals, by=0):
    """
    Sort a collection of notes by the specified attribute.

    Parameters
    ----------
    pitches : ndarray (N)
      Array of pitches corresponding to notes, sorted by selected attribute
      N - number of notes
    intervals : ndarray (N x 2)
      Array of onset-offset time pairs corresponding to notes, sorted by selected attribute
      N - number of notes
    by : int
      Index to sort notes by
      0 - onset | 1 - offset | 2 - pitch

    Returns
    ----------
    pitches : ndarray (N)
      Array of pitches corresponding to notes
      N - number of notes
    intervals : ndarray (N x 2)
      Array of onset-offset time pairs corresponding to notes
      N - number of notes
    """

    # Convert to batched notes for easy sorting
    batched_notes = notes_to_batched_notes(pitches, intervals)
    # Sort the batched notes
    batched_notes = sort_batched_notes(batched_notes, by)
    # Convert back to loose note groups
    pitches, intervals = batched_notes_to_notes(batched_notes)

    return pitches, intervals


def sort_pitch_list(times, pitch_list):
    """
    Sort a pitch list by frame time.

    Parameters
    ----------
    times : ndarray (N)
      Time in seconds of beginning of each frame
      N - number of time samples (frames)
    pitch_list : list of ndarray (N x [...])
      Array of pitches corresponding to notes
      N - number of pitch observations (frames)

    Returns
    ----------
    times : ndarray (N)
      Time in seconds of beginning of each frame, sorted by time
      N - number of time samples (frames)
    pitch_list : list of ndarray (N x [...])
      Array of pitches corresponding to notes
      N - number of pitch observations (frames), sorted by time
    """

    # Obtain the indices corresponding to the sorted times
    sort_order = list(np.argsort(times))

    # Sort the times
    times = np.sort(times)

    # Sort the pitch list
    pitch_list = [pitch_list[i] for i in sort_order]

    return times, pitch_list


##################################################
# DATA MANIPULATION                              #
##################################################


def rms_norm(audio):
    """
    Perform root-mean-square normalization.

    Parameters
    ----------
    audio : ndarray (N)
      Mono-channel audio to normalize
      N - number of samples in audio

    Returns
    ----------
    audio : ndarray (N)
      Normalized mono-channel audio
      N - number of samples in audio
    """

    # Calculate the square root of the squared mean
    rms = np.sqrt(np.mean(audio ** 2))

    # If root-mean-square is zero (audio is all zeros), do nothing
    if rms > 0:
        # Divide the audio by the root-mean-square
        audio = audio / rms

    return audio


def blur_activations(activations, kernel=None, normalize=False, threshold=False):
    """
    Blur activations by convolving them with a kernel.

    Parameters
    ----------
    activations : ndarray
      Provided activations
    kernel : list or ndarray
      Convolution kernel for blurring
    normalize : bool
      TODO - is this necessary? - is it too much to do outside in a separate call?
      Whether to normalize the activations after blurring
    threshold : bool
      TODO - is this necessary? - is it too much to do outside in a separate call?
      Whether to threshold the activations after blurring and (potentially) normalizing

    Returns
    ----------
    activations : ndarray
      Blurred activations
    """

    # Default the kernel to leave activations unchanged
    if kernel is None:
        kernel = [1]

    # Make sure the kernel is an ndarray (not a list)
    kernel = np.array(kernel)

    # Make sure the dimensionality matches
    if len(kernel.shape) != len(activations.shape):
        # Compute the number of dimensions missing from the kernel
        missing_dims = len(activations.shape) - len(kernel.shape)
        # Construct a matching shape for the kernel
        new_shape = tuple([1] * missing_dims) + tuple(kernel.shape)
        # Reshape the kernel
        kernel = np.reshape(kernel, new_shape)

    # Convolve the kernel with the activations
    activations = signal.convolve(activations, kernel, mode='same')

    if normalize:
        # Normalize with infinity norm
        activations = normalize_activations(activations)

    if threshold:
        # Threshold the activations (will remove pesky epsilons)
        activations = threshold_activations(activations)

    return activations


def normalize_activations(activations):
    """
    Normalizes an array of activations using infinity norm.

    Parameters
    ----------
    activations : ndarray
      Provided activations

    Returns
    ----------
    activations : ndarray
      Normalized activations
    """

    # Obtain the infinity norm of the activations
    inf_norm = np.max(np.abs(activations))

    # Avoid divide by zero
    if inf_norm != 0:
        # Divide the activations by the infinity norm
        activations = activations / inf_norm

    return activations


def threshold_activations(activations, threshold=0.5):
    """
    Performs binary thresholding on an array of activations.

    Parameters
    ----------
    activations : ndarray
      Provided activations
    threshold : float
      Value under which activations are negative

    Returns
    ----------
    activations : ndarray
      Thresholded activations
    """

    # Set all elements below threshold to zero (negative activation)
    activations[activations < threshold] = 0

    # Set remaining elements to one (positive activation)
    activations[activations != 0] = 1

    return activations


def framify_activations(activations, win_length, hop_length=1, pad=True):
    """
    Chunk activations into overlapping frames along the last dimension.

    Parameters
    ----------
    activations : ndarray
      Provided activations
    win_length : int
      Number of frames to include in each chunk
    hop_length : int
      Number of frames to skip between each chunk
    pad : bool
      Whether to pad incoming activations with zeros to give back array with same shape

    Returns
    ----------
    activations : ndarray
      Framified activations
    """

    # Determine the number of frames provided
    num_frames = activations.shape[-1]

    # Determine the pad length for the window (also used if not padding)
    pad_length = (win_length // 2)

    if pad:
        # Determine the number of frames required to yield same size
        num_frames_ = num_frames + 2 * pad_length
    else:
        # Make sure there are at least enough frames to fill one window
        num_frames_ = max(win_length, num_frames)

    # Pad the activations with zeros
    activations = librosa.util.pad_center(activations, size=num_frames_)

    # TODO - commented code is cleaner but breaks in PyTorch pipeline during model.pre_proc
    """
    # Convert the activations to a fortran array
    activations = np.asfortranarray(activations)

    # Framify the activations using librosa
    activations = librosa.util.frame(activations, frame_length=win_length, hop_length=hop_length).copy()

    # Switch window index and time index axes
    activations = np.swapaxes(activations, -1, -2)

    return activations
    """

    # Determine the number of hops in the activations
    num_hops = (num_frames_ - 2 * pad_length) // hop_length
    # Obtain the indices of the start of each chunk
    chunk_idcs = np.arange(0, num_hops) * hop_length

    # Chunk the activations with the specified window and hop length
    activations = [np.expand_dims(activations[..., i : i + win_length], axis=-2) for i in chunk_idcs]

    # Combine the chunks to get the framified activations
    activations = np.concatenate(activations, axis=-2)

    return activations


def inhibit_activations(activations, times, window_length):
    """
    Remove any activations within a specified time window following a previous activation.

    TODO - this is extremely slow for non-sparse activations

    Parameters
    ----------
    activations : ndarray
      Provided activations
    times : ndarray (N)
      Time in seconds of beginning of each frame
      N - number of time samples (frames)
    window_length : float
      Duration (seconds) of inhibition window

    Returns
    ----------
    activations : ndarray
      Inhibited activations
    """

    # Keep track of non-inhibited non-zeros
    pitch_idcs_keep = np.empty(0)
    frame_idcs_keep = np.empty(0)

    while True:
        # Determine the pitch and frame indices where activations begin
        pitch_idcs, frame_idcs = activations.nonzero()

        # Check if there are any non-zeros left to process
        if len(pitch_idcs) == 0 or len(frame_idcs) == 0:
            # If not, stop looping
            break

        # Determine the location of the next non-zero activation
        next_nz_pitch, next_nz_frame = pitch_idcs[0], frame_idcs[0]

        # Determine where the inhibition window ends
        inhibition_end = np.argmax(np.append(times, np.inf) >= times[next_nz_frame] + window_length)

        # Zero-out the activations in the inhibition window (including the non-zero itself)
        activations[next_nz_pitch, next_nz_frame : inhibition_end] = 0

        # The the non-zero that was just processed
        pitch_idcs_keep = np.append(pitch_idcs_keep, next_nz_pitch)
        frame_idcs_keep = np.append(frame_idcs_keep, next_nz_frame)

    # Add back in all of the non-inhibited non-zeros
    activations[pitch_idcs_keep.astype(constants.UINT),
                frame_idcs_keep.astype(constants.UINT)] = 1

    return activations


def remove_activation_blips(activations):
    """
    Remove blips (single-frame positives) in activations.

    Parameters
    ----------
    activations : ndarray
      Provided activations

    Returns
    ----------
    activations : ndarray
      Blip-free activations
    """

    # Determine where activations begin
    onsets = multi_pitch_to_onsets(activations)

    # Determine where activations end
    offsets = multi_pitch_to_offsets(activations)

    # Determine where the blips are located
    blip_locations = np.logical_and(onsets, offsets)

    # Zero out blips
    activations[blip_locations] = 0

    return activations


def interpolate_gaps(arr, gap_val=0):
    """
    Linearly interpolate between gaps in a one-dimensional array.

    Parameters
    ----------
    arr : ndarray
      One-dimensional array
    gap_val : float
      Value which indicates an empty entry

    Returns
    ----------
    arr : ndarray
      One-dimensional array
    """

    # Determine which entries occur directly before gaps
    gap_onsets = np.append(np.diff((arr == gap_val).astype(constants.INT)), [0]) == 1
    # Determine which entries occur directly after gaps
    gap_offsets = np.append([0], np.diff(np.logical_not(arr == gap_val).astype(constants.INT))) == 1
    # Identify any onset and offset indices
    onset_idcs, offset_idcs = np.where(gap_onsets)[0], np.where(gap_offsets)[0]

    # Determine where the first onset and last offset occur
    first_onset = np.min(onset_idcs) if len(onset_idcs) else len(arr)
    last_offset = np.max(offset_idcs) if len(offset_idcs) else 0

    # Ignore offsets before first onset
    offset_idcs = offset_idcs[offset_idcs > first_onset]
    # Ignore onsets after last offset
    onset_idcs = onset_idcs[onset_idcs < last_offset]

    # Construct a list of the indices surrounding gaps
    gap_idcs = np.array([onset_idcs, offset_idcs]).T

    # Loop through all gaps in the array
    for start, end in [list(gap) for gap in gap_idcs]:
        # Extract the gap boundaries to interpolate between
        interp_start, interp_stop = arr[start], arr[end]
        # Determine the number of values for the interpolation
        num_values = end - start + 1
        # Linearly interpolate across frames with empty pitch observations
        arr[start: end + 1] = np.linspace(interp_start, interp_stop, num_values)

    return arr


def get_resample_idcs(times, target_times):
    """
    Obtain indices to resample from a set of original times to
    a set of target times using nearest neighbor interpolation.

    Parameters
    ----------
    times : ndarray (N)
      Array of original times
    target_times : ndarray (K)
      Array of target times

    Returns
    ----------
    resample_idcs : ndarray (K)
      Indices corresponding to the original times nearest to the target times at each step
    """

    # Determine how many original and target times were given
    num_times = len(times)
    num_targets = len(target_times)

    if not num_times:
        # No original times exist, there is nothing to
        # index and therefore no indices are returned
        resample_idcs = None
    elif not num_targets:
        # No target times exist, so the indices would
        # correspond to an empty array (of indices)
        resample_idcs = np.empty(0, dtype=constants.INT)
    else:
        # Create an array of indices pointing to all original entries
        original_idcs = np.arange(0, num_times)

        # Clamp resampled indices within the valid range
        fill_values = (original_idcs[0], original_idcs[-1])

        # Obtain the indices to resample from the original to
        # the target times using nearest neighbor interpolation
        resample_idcs = scipy.interpolate.interp1d(times, original_idcs,
                                                   kind='nearest',
                                                   bounds_error=False,
                                                   fill_value=fill_values,
                                                   assume_sorted=True)(target_times).astype(constants.INT)

    return resample_idcs


##################################################
# UTILITY                                        #
##################################################


def seed_everything(seed):
    """
    Set all necessary seeds for PyTorch at once.


    WARNING: the number of workers in the training loader affects behavior:
             this is because each sample will inevitably end up being processed
             by a different worker if num_workers is changed, and each worker
             has its own random seed
             TODO - I will fix this in the future if possible

    Parameters
    ----------
    seed : int
      Seed to use for random number generation
    """

    # TODO - look into 'torch.backends.cudnn.benchmark = False'
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def estimate_hop_length(times):
    """
    Estimate hop length of a semi-regular but non-uniform series of times.

    Adapted from mir_eval pull request #336.

    Parameters
    ----------
    times : ndarray
      Array of times corresponding to a time series

    Returns
    ----------
    hop_length : float
      Estimated hop length (seconds)
    """

    if not len(times):
        raise ValueError('Cannot estimate hop length from an empty time array.')

    # Make sure the times are sorted
    times = np.sort(times)

    # Determine where there are no gaps
    non_gaps = np.append([False], np.isclose(np.diff(times, n=2), 0))

    if not np.sum(non_gaps):
        raise ValueError('Time observations are too irregular.')

    # Take the median of the time differences at non-gaps
    hop_length = np.median(np.diff(times)[non_gaps])

    return hop_length


def time_series_to_uniform(times, values, hop_length=None, duration=None, suppress_warnings=True):
    """
    Convert a semi-regular time series with gaps into a uniform time series.

    Adapted from mir_eval pull request #336.

    Parameters
    ----------
    times : ndarray
      Array of times corresponding to a time series
    values : list of ndarray
      Observations made at times
    hop_length : number or None (optional)
      Time interval (seconds) between each observation in the uniform series
    duration : number or None (optional)
      Total length (seconds) of times series
      If specified, should be greater than all observation times
    suppress_warnings : bool
      Whether to ignore warning messages

    Returns
    -------
    times : ndarray
      Uniform time array
    values : ndarray
      Observations corresponding to uniform times
    """

    if not len(times) or not len(values):
        return np.array([]), []

    if hop_length is None:
        if not suppress_warnings:
            warnings.warn('Since hop length is unknown, it will be estimated. ' +
                          'This may lead to unwanted behavior if the observation ' +
                          'times are sporadic or irregular.', category=RuntimeWarning)

        # Estimate the hop length if it was not provided
        hop_length = estimate_hop_length(times)

    if duration is None:
        # Default the duration to the last reported time in the series
        duration = times[-1]

    # Determine the total number of observations in the uniform time series
    num_entries = int(np.ceil(duration / hop_length)) + 1

    # Attempt to fill in blank frames with the appropriate value
    empty_fill = np.array([])
    new_values = [empty_fill] * num_entries
    new_times = hop_length * np.arange(num_entries)

    # Determine which indices the provided observations fall under
    idcs = np.round(times / hop_length).astype(int)

    # Fill the observed values into their respective locations in the uniform series
    for i in range(len(idcs)):
        if times[i] <= duration:
            new_values[idcs[i]] = values[i]

    return new_times, new_values


def get_frame_times(duration, sample_rate, hop_length):
    """
    Determine the start time of each frame for given audio parameters

    Parameters
    ----------
    duration : float
        Total length (seconds) of audio
    sample_rate : int or float
        Number of samples per second
    hop_length : int or float
        Number of samples between frames

    Returns
    -------
    times : ndarray
        Array of times corresponding to frames
    """

    # TODO - this seems too close to the FeatureModel function -
    #        maybe that function should be made to work with a duration if no audio exists

    # Determine the total number of frames in the sample
    total_num_frames = int(1 + (duration * sample_rate - 1) // hop_length)

    # We need the frame times for the tablature
    times = librosa.frames_to_time(np.arange(total_num_frames), sr=sample_rate, hop_length=hop_length)

    return times


def apply_func_stacked_representation(stacked_representation, func, **kwargs):
    """
    Recursively apply a function to the contents of each slice in a stacked representation.
    TODO - this can be probably be used in many places to avoid extra code

    Parameters
    ----------
    stacked_representation : dict
        Dictionary representing some stacked data structure
    func : function
        Function to run on each slice
    kwargs : dict of keyword arguments
        Arguments for the chosen function

    Returns
    -------
    stacked_representation : dict
        Dictionary representing some modified stacked data structure
    """

    # Make a copy of the stacked representation
    stacked_representation = deepcopy(stacked_representation)

    # Loop through the stack
    for slc in stacked_representation.keys():
        # Use the current slice contents as function arguments
        args = stacked_representation[slc]

        # Check if there is more than one argument
        if isinstance(args, tuple):
            # Unpack the arguments and run the function
            output = func(*args, **kwargs)
        else:
            # Run the function with the single argument
            output = func(args, **kwargs)

        # Apply the given function
        stacked_representation[slc] = output

    return stacked_representation


def pack_stacked_representation(stacked_representation):
    """
    Package together the key-value pairs of a stacked representation in a save-friendly format.

    Parameters
    ----------
    stacked_representation : dict
      Dictionary containing (slice -> (...)) pairs

    Returns
    ----------
    packed_stacked_representation : ndarray (S x 2)
      Stacked representation packaged as object ndarray
      S - number of slices in the stack
    """

    # Concatenate the key-value pairs of each slice and wrap with object dtype
    packed_stacked_representation = np.array(list(stacked_representation.items()), dtype=object)

    return packed_stacked_representation


def unpack_stacked_representation(packed_stacked_representation):
    """
    Unpack from save-friendly format the key-value pairs of a stacked representation.

    Parameters
    ----------
    packed_stacked_representation : ndarray (S x 2)
      Stacked representation packaged in object ndarray
      S - number of slices in the stack

    Returns
    ----------
    stacked_representation : dict
      Dictionary containing (slice -> (...)) pairs
    """

    # Simply treat each array along the first dimension as a separate key-value pair
    stacked_representation = dict(packed_stacked_representation)

    return stacked_representation


def tensor_to_array(data):
    """
    Simple helper function to convert a PyTorch tensor
    into a NumPy array in order to keep code readable.

    Parameters
    ----------
    data : PyTorch tensor
      Tensor to convert to array

    Returns
    ----------
    array : NumPy ndarray
      Converted array
    """

    # Make sure the input is a PyTorch tensor
    if isinstance(data, torch.Tensor):
        # Change device to CPU,
        # detach from gradient graph,
        # and convert to NumPy array
        data = data.cpu().detach().numpy()

    return data


def array_to_tensor(data, device=None):
    """
    Simple helper function to convert a NumPy array
    into a PyTorch tensor in order to keep code readable.

    Parameters
    ----------
    data : NumPy ndarray
      Array to convert to tensor
    device : string, or None (optional)
      Add tensor to this device, if specified

    Returns
    ----------
    tensor : PyTorch tensor
      Converted tensor
    """

    # Make sure the input is a NumPy array
    if isinstance(data, np.ndarray):
        # Convert to PyTorch tensor
        data = torch.from_numpy(data)

        # Add tensor to device, if specified
        if device is not None:
            data = data.to(device)

    return data


def save_dict_npz(path, d):
    """
    Simple helper function to save a dictionary as a compressed NumPy zip file.

    Parameters
    ----------
    path : string
      Path to save the NumPy zip file
    d : dict
      Dictionary of entries to save
    """

    # Save the dictionary as a NumPy zip at the specified path
    np.savez_compressed(path, **d)


def load_dict_npz(path):
    """
    Simple helper function to load a NumPy zip file.

    Parameters
    ----------
    path : string
      Path to load the NumPy zip file

    Returns
    ----------
    data : dict
      Unpacked dictionary
    """

    # Load the NumPy zip file at the path
    data = dict(np.load(path, allow_pickle=True))

    return data


def dict_to_dtype(track, dtype):
    """
    Convert all ndarray entries in a dictionary to a specified type.

    Parameters
    ----------
    track : dict
      Dictionary containing data for a track
    dtype : string or type
      TODO - will type work?
      Ndarray dtype to convert

    Returns
    ----------
    track : dict
      Dictionary containing data for a track
    """

    # Copy the dictionary to avoid hard assignment
    track = deepcopy(track)

    # Obtain a list of the dictionary keys
    keys = list(track.keys())

    # Loop through the dictionary keys
    for key in keys:
        # Check if the entry is another dictionary
        if isinstance(track[key], dict):
            # Call this function recursively
            track[key] = dict_to_dtype(track[key], dtype)
        # Check if the dictionary entry is an ndarray
        elif isinstance(track[key], np.ndarray):
            # Convert the ndarray to the specified type
            track[key] = track[key].astype(dtype)

        # TODO - convert non ndarray to similar type?
        #if isinstance(track[key], int):
        #    track[key] = float(track[key])

    return track


def dict_to_device(track, device):
    """
    Add all tensor entries in a dictionary to a specified device.

    Parameters
    ----------
    track : dict
      Dictionary containing data for a track
    device : string, or None (optional)
      Add tensor to this device, if specified

    Returns
    ----------
    track : dict
      Dictionary containing data for a track
    """

    # Copy the dictionary to avoid hard assignment
    track = deepcopy(track)

    # Obtain a list of the dictionary keys
    keys = list(track.keys())

    # Loop through the dictionary keys
    for key in keys:
        # Check if the entry is another dictionary
        if isinstance(track[key], dict):
            # Call this function recursively
            track[key] = dict_to_device(track[key], device)
        # Check if the dictionary entry is a tensor
        elif isinstance(track[key], torch.Tensor):
            # Add the tensor to the specified device
            track[key] = track[key].to(device)

    return track


def dict_to_array(track):
    """
    Convert all tensor entries in a dictionary to ndarray.

    Parameters
    ----------
    track : dict
      Dictionary containing data for a track

    Returns
    ----------
    track : dict
      Dictionary containing data for a track
    """

    # Copy the dictionary to avoid hard assignment
    # TODO - can't copy tensors with gradients
    #track = deepcopy(track)

    # Obtain a list of the dictionary keys
    keys = list(track.keys())

    # Loop through the dictionary keys
    for key in keys:
        # Check if the entry is another dictionary
        if isinstance(track[key], dict):
            # Call this function recursively
            track[key] = dict_to_array(track[key])
        # Check if the entry is a tensor
        elif isinstance(track[key], torch.Tensor):
            # Convert the tensor to an array
            track[key] = tensor_to_array(track[key])

    return track


def dict_to_tensor(track):
    """
    Convert all ndarray entries in a dictionary to tensors.

    Parameters
    ----------
    track : dict
      Dictionary containing data for a track

    Returns
    ----------
    track : dict
      Dictionary containing data for a track
    """

    # Copy the dictionary to avoid hard assignment
    track = deepcopy(track)

    # Obtain a list of the dictionary keys
    keys = list(track.keys())

    # Loop through the dictionary keys
    for key in keys:
        # Check if the entry is another dictionary
        if isinstance(track[key], dict):
            # Call this function recursively
            track[key] = dict_to_tensor(track[key])
        # Check if the entry is an array
        elif isinstance(track[key], np.ndarray):
            # Convert the array to a tensor
            track[key] = array_to_tensor(track[key])

    return track


def dict_squeeze(track, dim=None):
    """
    Collapse unnecessary dimensions of an array or tensor.

    Parameters
    ----------
    track : dict
      Dictionary containing data for a track
    dim : int or None
      Dimension to collapse (any single dimensions if unspecified)

    Returns
    ----------
    track : dict
      Dictionary containing data for a track
    """

    # Copy the dictionary to avoid hard assignment
    # TODO - can't copy tensors with gradients
    #track = deepcopy(track)

    # Obtain a list of the dictionary keys
    keys = list(track.keys())

    # Loop through the dictionary keys
    for key in keys:
        # Check if the entry is another dictionary
        if isinstance(track[key], dict):
            # Call this function recursively
            track[key] = dict_squeeze(track[key])
        # Check if the entry is a tensor or array
        elif isinstance(track[key], torch.Tensor) or isinstance(track[key], np.ndarray):
            if dim is None:
                # Squeeze all unnecessary dimensions of the tensor or array
                track[key] = track[key].squeeze()
            elif track[key].shape[dim] == 1:
                # Squeeze the chosen dimension of the tensor or array
                track[key] = track[key].squeeze(dim)

    return track


def dict_unsqueeze(track, dim=0):
    """
    Add a new dimension to an array or tensor.

    Parameters
    ----------
    track : dict
      Dictionary containing data for a track
    dim : int
      Insertion point of new dimension

    Returns
    ----------
    track : dict
      Dictionary containing data for a track
    """

    # Copy the dictionary to avoid hard assignment
    track = deepcopy(track)

    # Obtain a list of the dictionary keys
    keys = list(track.keys())

    # Loop through the dictionary keys
    for key in keys:
        # Check if the entry is another dictionary
        if isinstance(track[key], dict):
            # Call this function recursively
            track[key] = dict_unsqueeze(track[key])
        # Check if the dictionary entry is a Tensor
        elif isinstance(track[key], torch.Tensor):
            # Add a new dimension at the insertion point
            track[key] = track[key].unsqueeze(dim)
        # Check if the entry is an ndarray
        elif isinstance(track[key], np.ndarray):
            # Add a new dimension at the insertion point
            track[key] = np.expand_dims(track[key], axis=dim)

    return track


def dict_append(track, additions, dim=-1):
    """
    Append together matching entries of two dictionaries. This
    will deliberately skip tuples, as in stacked representations.
    TODO - may be repeat of function defined in evaluate.py

    Parameters
    ----------
    track : dict
      Dictionary containing data for a track
    additions : dict
      Dictionary containing new data
    dim : int
      Dimension on which to append

    Returns
    ----------
    track : dict
      Dictionary containing data for a track
    """

    # Copy the dictionary to avoid hard assignment
    track = deepcopy(track)

    # Obtain a list of the dictionary keys
    keys = list(additions.keys())

    # Loop through the dictionary keys
    for key in keys:
        # Check if the entry exists
        if key not in track:
            # Add the new entry to the current dictionary
            track[key] = additions[key]
        # Check if the entry is another dictionary
        elif isinstance(track[key], dict):
            # Call this function recursively
            track[key] = dict_append(track[key], additions[key], dim)
        # Check if the entry is a list
        elif isinstance(additions[key], list):
            # Add the contents of the lists together
            track[key] += additions[key]
        # Check if the dictionary entry is an ndarray
        elif isinstance(additions[key], np.ndarray):
            # Use the NumPy append function
            track[key] = np.append(track[key], additions[key], axis=dim)
        # Check if the dictionary entry is a tensor
        elif isinstance(additions[key], torch.Tensor):
            # Use the Torch cat function
            track[key] = torch.cat((track[key], additions[key]), dim=dim)
        # Check if the dictionary entry is a tuple
        elif isinstance(additions[key], tuple):
            # Insert a None to show we saw the tuple but refuse to process it
            track[key] = None

    return track


def dict_detach(track):
    """
    Detach gradient computation for all tensors.

    Parameters
    ----------
    track : dict
      Dictionary containing data for a track

    Returns
    ----------
    track : dict
      Dictionary containing data for a track
    """

    # Obtain a list of the dictionary keys
    keys = list(track.keys())

    # Loop through the dictionary keys
    for key in keys:
        # Check if the dictionary entry is a Tensor
        if isinstance(track[key], torch.Tensor):
            # Detach the gradient from the Tensor
            track[key] = track[key].detach()

    return track


def unpack_dict(data, key):
    """
    Determine the corresponding entry for a dictionary key.

    TODO - can use this many places (e.g. datasets) to be safe and neat

    Parameters
    ----------
    data : dictionary or object
      Object to query as being a dictionary and containing the specified key
    key : string
      Key specifying entry to unpack, if possible

    Returns
    ----------
    entry : object or None
      Unpacked entry or None to indicate non-existence
    """

    # Default the entry
    # TODO - better to use None or False?
    # TODO - can give me pointer to actual item, not copy
    entry = None

    # Check if a dictionary was provided and if the key is in the dictionary
    if isinstance(data, dict) and query_dict(data, key):
        # Unpack the relevant entry
        # TODO - return a copy?
        entry = data[key]

    return entry


def query_dict(dictionary, key):
    """
    Determine if a dictionary has an entry for a specified key.

    TODO - can use this many places (e.g. datasets) to be safe and neat

    Parameters
    ----------
    dictionary : dict
      Dictionary to query
    key : string
      Key to query

    Returns
    ----------
    exists : bool
      Whether or not the key exists in the dictionary
    """

    # Check if the dictionary contains the key
    exists = key in dictionary.keys()

    return exists


def get_tag(tag=None):
    """
    Simple helper function to create a tag for saving a file if one does not already exist.

    This is useful because in some places we don't know whether we will have a tag or not,
    but still want to save files.

    Parameters
    ----------
    tag : string or None (optional)
      Name of file if it already exists

    Returns
    ----------
    tag : string or None (optional)
      Name picked for the file
    """

    # Get the data and time in a file-saving--friendly format
    date_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

    # If there is no tag, use the date and time
    tag = date_time if tag is None else tag

    return tag


def slice_track(track, start, stop, skip=None, pad=True):
    """
    Slice any ndarray or tensor entries of a dictionary along the last axis.

    Parameters
    ----------
    track : dict
      Dictionary containing data for a track
    start : int
      Beginning index
    stop : int
      End index (excluded in slice)
    skip : list of str
      Keys to skip during this process
    pad : bool
      Whether to pad to implicit size (stop - start) when necessary

    Returns
    ----------
    track : dict
      Dictionary containing data for a track
    """

    # Default the skipped keys to an empty list if unspecified
    if skip is None:
        skip = list()

    # Copy the dictionary to avoid hard assignment
    track = deepcopy(track)

    # Obtain a list of the dictionary keys
    keys = list(track.keys())

    # Loop through the dictionary keys
    for key in keys:
        # Check if the dictionary entry is an ndarray or tensor
        if key not in skip and (isinstance(track[key], np.ndarray) or
                                isinstance(track[key], torch.Tensor)):
            # Slice along the final axis
            track[key] = track[key][..., start : stop]

            # Determine if the entry was long enough
            num_missing = max(0, (stop - start) - track[key].shape[-1])

            if num_missing:
                # Create an array or tensor of zeros to add to the entry
                if isinstance(track[key], np.ndarray):
                    # Append a NumPy array of zeros
                    zeros = np.zeros(track[key].shape[:-1] + tuple([num_missing]))
                    track[key] = np.concatenate((track[key], zeros), axis=-1)
                else:
                    # Append a PyTorch tensor of zeros
                    zeros = torch.zeros(track[key].shape[:-1] + tuple([num_missing])).to(track[key].device)
                    track[key] = torch.cat((track[key], zeros), dim=-1)

                if key == constants.KEY_TABLATURE:
                    # Change the padded zeros to ones
                    track[key][..., -num_missing:] = -1

    return track


def get_current_time(decimals=3):
    """
    Determine the current system time.

    Parameters
    ----------
    decimals : int
      Number of digits to keep when rounding

    Returns
    ----------
    current_time : float
      Current system time
    """

    # Get the current time and round to the specified number of digits
    current_time = round(time.time(), decimals)

    return current_time


def print_time(t, label=None):
    """
    Print a time to the console.

    Parameters
    ----------
    t : float
      Arbitrary time
    label : string or None (Optional)
      Label for the time print statement
    """

    # Begin constructing the string
    string = 'Time'

    if label is not None:
        # Add the label if it was specified
        string += f' ({label})'

    # Add the time to the string
    string += f' : {t}'

    # Print the constructed string
    print(string)


def compute_time_difference(start_time, pr=True, label=None, decimals=3):
    """
    Obtain the time elapsed since the given system time.

    Parameters
    ----------
    start_time : float
      Arbitrary system time
    decimals : int
      Number of digits to keep when rounding
    pr : bool
      Whether to print the time difference to the console
    label : string or None (Optional)
      Label for the optional print statement

    Returns
    ----------
    elapsed_time : float
      Time elapsed since specified time
    """

    # Take the difference between the current time and the paramterized time
    elapsed_time = round(get_current_time(decimals) - start_time, decimals)

    if pr:
        # Print to console
        print_time(elapsed_time, label)

    return elapsed_time
