# My imports
import amt_models.tools.constants as constants

# Regular imports
from copy import deepcopy

import numpy as np
import mir_eval
import librosa
import random
import torch


def stacked_notes_to_notes(stacked_notes):
    """
    Convert a dictionary of stacked notes into a single representation.

    Parameters
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs

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

    # Sort the notes by onset
    pitches, intervals = sort_notes(pitches, intervals)

    return pitches, intervals


def notes_to_stacked_notes(pitches, intervals, i=0):
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
    i : int
      Slice key to use

    Returns
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    """

    # Initialize a dictionary to hold the notes
    stacked_notes = dict()

    # Add the pitch-interval pairs to the stacked notes dictionary under the slice key
    stacked_notes[i] = sort_notes(pitches, intervals)

    return stacked_notes


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
      Array of note pitches and intervals by row
      N - number of notes
    """

    # Default the batched notes to an empty array of the correct shape
    batched_notes = np.empty([0, 3])

    if len(pitches) > 0:
        # Add an extra dimension to the pitches to match dimensionality of intervals
        pitches = np.expand_dims(pitches, axis=-1)
        # Concatenate the loose arrays to obtain ndarray([[onset, offset, pitch]])
        batched_notes = np.concatenate((intervals, pitches), axis=-1)

    # TODO - validation check?

    return batched_notes


def batched_notes_to_notes(batched_notes):
    """
    Convert batch-friendly notes into loose note groups.

    Parameters
    ----------
    batched_notes : ndarray (N x 3)
      Array of note pitches and intervals by row
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

    # TODO - validation check?

    return pitches, intervals


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

    # Define the attributes that can be used to sort the notes
    attributes = ['onset', 'offset', 'pitch']

    # Obtain the dtype of the batch-friendly notes before any manipulation
    dtype = batched_notes.dtype
    # Set a temporary dtype for sorting purposes
    batched_notes.dtype = [(attributes[0], float), (attributes[1], float), (attributes[2], float)]
    # Sort the notes along the row axis by the selected attribute
    batched_notes = np.sort(batched_notes, axis=0, order=attributes[by])
    # Reset the dtype of the batch-friendly notes
    batched_notes.dtype = dtype

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

    # Collapse the times from each pitch_list into one array
    times = np.unique(np.concatenate([pair[0] for pair in pitch_list_pairs]))

    # Initialize empty pitch arrays for each time entry
    pitch_list = [np.empty(0)] * times.size

    # Loop through each pitch list
    for slice_times, slice_pitch_arrays in pitch_list_pairs:
        # Loop through the pitch list entries
        for entry in range(len(slice_pitch_arrays)):
            # Determine where this entry belongs in the new pitch list
            idx = np.where(times == slice_times[entry])[0].item()
            # Insert the frequencies at the corresponding time
            pitch_list[idx] = np.append(pitch_list[idx], slice_pitch_arrays[entry])

    # Sort the time-pitch array pairs by time
    times, pitch_list = sort_pitch_list(times, pitch_list)

    return times, pitch_list


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


def stacked_notes_to_stacked_multi_pitch(stacked_notes, times, profile):
    """
    Convert a dictionary of note groups into a stack of multi pitch arrays.

    Parameters
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    times : ndarray (N)
      Time in seconds of beginning of each frame
      N - number of time samples (frames)
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
    for slc in range(len(stacked_notes)):
        # Get the pitches and intervals from the slice
        pitches, intervals = stacked_notes[slc]
        multi_pitch = notes_to_multi_pitch(pitches, intervals, times, profile)
        stacked_multi_pitch.append(np.expand_dims(multi_pitch, axis=0))

    # Collapse the list into an array
    stacked_multi_pitch = np.concatenate(stacked_multi_pitch)

    return stacked_multi_pitch


def notes_to_multi_pitch(pitches, intervals, times, profile):
    """
    Convert loose note groups into a multi pitch array.

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

    Returns
    ----------
    multi_pitch : ndarray (F x T)
      Discrete pitch activation map
      F - number of discrete pitches
      T - number of frames
    """

    # Determine the dimensionality of the multi pitch array
    num_pitches = profile.get_range_len()
    num_frames = len(times) - 1

    # Initialize an empty multi pitch array
    multi_pitch = np.zeros((num_pitches, num_frames))

    # Convert the pitches to number of semitones from lowest note
    pitches = np.round(pitches - profile.low).astype('uint')

    # Duplicate the array of times for each note and stack along a new axis
    times = np.concatenate([[times]] * max(1, len(pitches)), axis=0)

    # Determine the frame where each note begins and ends
    onsets = np.argmin((times <= intervals[..., :1]), axis=1) - 1
    offsets = np.argmin((times < intervals[..., 1:]), axis=1) - 1

    # Loop through each note
    for i in range(len(pitches)):
        # Populate the multi pitch array with activations for the note
        multi_pitch[pitches[i], onsets[i] : offsets[i] + 1] = 1

    return multi_pitch


def stacked_pitch_list_to_stacked_multi_pitch(stacked_pitch_list, profile):
    """
    Convert a stacked pitch list into a stack of multi pitch arrays.

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
    for slc in range(len(stacked_pitch_list)):
        # Get the pitches and intervals from the slice
        times, pitch_list = stacked_pitch_list[slc]
        multi_pitch = pitch_list_to_multi_pitch(times, pitch_list, profile)
        stacked_multi_pitch.append(np.expand_dims(multi_pitch, axis=0))

    # Collapse the list into an array
    stacked_multi_pitch = np.concatenate(stacked_multi_pitch)

    return stacked_multi_pitch


def pitch_list_to_multi_pitch(times, pitch_list, profile, tolerance=0.5):
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
    profile : InstrumentProfile (instrument.py)
      Instrument profile detailing experimental setup
    tolerance : float
      Amount of semitone deviation allowed

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

    # Loop through each note
    for i in range(len(pitch_list)):
        # Calculate the pitch semitone difference from the lowest note
        difference = pitch_list[i] - profile.low
        # Determine the amount of semitone deviation for each pitch
        deviation = difference % 1
        deviation[deviation > 0.5] -= 1
        deviation = np.abs(deviation)
        # Convert the pitches to number of semitones from lowest note
        pitches = np.round(difference[deviation < tolerance]).astype('uint')
        # Populate the multi pitch array with activations
        multi_pitch[pitches, i] = 1

    return multi_pitch


# TODO - major cleanup needed for all of these functions


def tabs_to_multi_pianoroll(tabs, profile):
    # TODO - for now make sure it's guitar - can generalize in future
    #assert isinstance(profile, GuitarProfile)

    tabs = tabs.copy().astype('int')

    shape = list(tabs.shape) + [profile.get_range_len()]
    #shape = list(tabs.shape)
    #shape = shape[:-1] + [profile.get_range_len()] + shape[-1:]
    pianoroll = np.zeros(shape)

    midi_tuning = profile.get_midi_tuning()
    multi_start = midi_tuning - profile.low

    if shape[0] != profile.num_strings:
        # Add a batch dimension
        multi_start = np.expand_dims(multi_start, axis=0)
        multi_start = np.repeat(multi_start, shape[0], axis=0)

    non_silent = tabs != -1

    pitches = tabs + multi_start
    pitches = pitches[non_silent]

    idcs = tuple(list(non_silent.nonzero()) + [pitches])

    pianoroll[idcs] = 1
    pianoroll = np.swapaxes(pianoroll, -1, -2)

    return pianoroll


def tabs_to_pianoroll(tabs):
    pianoroll = tabs_to_multi_pianoroll(tabs)

    pianoroll = np.max(pianoroll, axis=0)

    return pianoroll


def get_onsets(pitch, profile):
    to_torch = 'torch' in str(pitch.dtype)
    if to_torch:
        device = pitch.device
        pitch = pitch.cpu().detach().numpy()

    tabs = False
    onsets = None

    # TODO - how to deal with batch dimension properly?
    if valid_tabs(pitch[0], profile):
        pitch = tabs_to_multi_pianoroll(pitch, profile)
        tabs = True

    if valid_multi(pitch[0], profile) or valid_single(pitch[0], profile):
        onsets = get_pianoroll_onsets(pitch)

    if tabs:
        onsets = multi_pianoroll_to_tabs(onsets, profile)

    if to_torch:
        onsets = torch.from_numpy(onsets)
        onsets = onsets.to(device)

    return onsets


def multi_pianoroll_to_tabs(multi_pianoroll, profile):
    # TODO - for now make sure it's guitar - can generalize (num_dofs) in future
    #assert isinstance(profile, GuitarProfile)

    shape = list(multi_pianoroll.shape)
    silent_idx = shape.pop(-2)

    no_note_row = np.ones(shape)
    no_note_row = np.expand_dims(no_note_row, axis=-2)
    multi_pianoroll = np.append(multi_pianoroll, no_note_row, axis=-2)

    tabs = np.argmax(multi_pianoroll, axis=-2)

    silent = tabs == silent_idx

    midi_tuning = profile.get_midi_tuning()
    multi_start = midi_tuning - profile.low

    if shape[0] != profile.num_strings:
        # Add a batch dimension
        multi_start = np.expand_dims(multi_start, axis=0)
        multi_start = np.repeat(multi_start, shape[0], axis=0)

    tabs = tabs - multi_start

    tabs[silent] = -1
    return tabs


def get_pianoroll_onsets(pianoroll):
    first_frame = pianoroll[..., :1]
    adjacent_diff = pianoroll[..., 1:] - pianoroll[..., :-1]
    onsets = np.concatenate([first_frame, adjacent_diff], axis=-1) == 1
    return onsets


def get_pianoroll_offsets(pianoroll):
    pass


def get_note_offsets(note_arr):
    pass


def pianoroll_to_pitchlist(pianoroll, lowest):
    active_pitches = []

    # TODO - vectorize?
    for i in range(pianoroll.shape[-1]): # For each frame
        # Determine the activations across this frame
        active_pitches += [librosa.midi_to_hz(np.where(pianoroll[:, i] != 0)[0] + lowest)]

    return active_pitches


def to_single(activations, profile):
    if valid_tabs(activations, profile):
        activations = tabs_to_multi_pianoroll(activations, profile)

    if valid_multi(activations, profile):
        # TODO - multi_to_single and single_to_multi funcs
        activations = np.max(activations, axis=0)

    assert valid_single(activations, profile)

    return activations


def to_multi(activations, profile):
    if valid_single(activations, profile):
        activations = np.expand_dims(activations, axis=0)

    if valid_tabs(activations, profile):
        activations = tabs_to_multi_pianoroll(activations, profile)

    assert valid_multi(activations, profile)

    return activations


def to_tabs(activations, profile):
    if valid_single(activations, profile):
        activations = np.expand_dims(activations, axis=0)

    if valid_multi(activations, profile):
        activations = multi_pianoroll_to_tabs(activations, profile)

    assert valid_tabs(activations, profile)

    return activations


def feats_to_batch(feats, times):
    # TODO - a function which accepts only feats (for deployment)
    pass


def track_to_batch(track):
    batch = deepcopy(track)

    if 'track' in batch:
        batch['track'] = [batch['track']]
    else:
        batch['track'] = ['no_name']

    keys = list(batch.keys())
    for key in keys:
        if isinstance(batch[key], np.ndarray):
            batch[key] = torch.from_numpy(batch[key]).unsqueeze(0)

    return batch


def track_to_dtype(track, dtype='float32'):
    track = deepcopy(track)

    keys = list(track.keys())
    for key in keys:
        if isinstance(track[key], np.ndarray):
            track[key] = track[key].astype(dtype)
        if isinstance(track[key], int):
            track[key] = float(track[key])

    return track


def track_to_device(track, device):
    keys = list(track.keys())

    for key in keys:
        if isinstance(track[key], torch.Tensor):
            track[key] = track[key].to(device)

    return track


def track_to_cpu(track):
    keys = list(track.keys())

    for key in keys:
        if isinstance(track[key], torch.Tensor):
            track[key] = track[key].squeeze().cpu().detach().numpy()

    return track


def seed_everything(seed):
    # WARNING: the number of workers in the training loader affects behavior:
    #          this is because each sample will inevitably end up being processed
    #          by a different worker if num_workers is changed, and each worker
    #          has its own random seed
    #          TODO - I will fix this in the future if possible

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


# TODO - should I be using a yield function instead of for batch in loader?


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


def framify_tfr(tfr, win_length, hop_length, pad=None):
    # TODO - avoid conversion in collate_fn instead?
    to_torch = False
    if 'torch' in str(tfr.dtype):
        to_torch = True
        tfr = tfr.cpu().detach().numpy()

    # TODO - parameterize axis or just assume -1?
    if pad is not None:
        # TODO - librosa pad center?
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


"""
def infer_lowest_note(pianoroll):
    note_range = pianoroll.shape[0]
    if note_range == PIANO_RANGE:
        return PIANO_LOWEST
    elif note_range == GUITAR_RANGE:
        return GUITAR_LOWEST
    else:
        # Something went awry
        return None
"""


def threshold_arr(arr, thr):
    arr[arr < thr] = 0
    arr[arr != 0] = 1
    return arr


def valid_activations(activations, profile):
    # TODO - add valid pitchlist?
    valid = valid_single(activations, profile)
    valid = valid or valid_multi(activations, profile)
    valid = valid or valid_tabs(activations, profile)

    return valid


def valid_single(activations, profile):
    single = True

    shape = activations.shape

    if len(shape) != 2:
        single = False

    if shape[0] != profile.get_range_len():
        single = False

    if isinstance(activations, np.ndarray):
        if np.max(activations) > 1:
            single = False

        if np.min(activations) < 0:
            single = False

    if isinstance(activations, torch.Tensor):
        if torch.max(activations) > 1:
            single = False

        if torch.min(activations) < 0:
            single = False

    return single


def valid_multi(activations, profile):
    multi = True

    shape = activations.shape

    if len(shape) != 3:
        multi = False

    if shape[1] != profile.get_range_len():
        multi = False

    return multi


def valid_tabs(activations, profile):
    tabs = True

    shape = activations.shape

    if len(shape) != 2:
        tabs = False

    if shape[0] == profile.get_range_len():
        tabs = False

    # Must be exact integers
    if isinstance(activations, np.ndarray):
        if np.sum(activations - np.round(activations)) != 0:
            tabs = False

    if isinstance(activations, torch.Tensor):
        if torch.sum(activations - torch.round(activations)) != 0:
            tabs = False

    return tabs


def valid_notes(pitches, intervals):
    # TODO - array dimensions as well
    # Validate the intervals
    valid = librosa.util.valid_intervals(intervals)

    # Validate the pitches - should be in Hz
    try:
        mir_eval.util.validate_frequencies(pitches, 5000, 20)
    except ValueError:
        valid = False

    return valid


# TODO - use this standardized version everywhere
def get_batch_size(batch):
    if isinstance(batch, dict):
        bs = len(batch['track'])
    elif isinstance(batch, np.ndarray) or isinstance(batch, torch.Tensor):
        bs = batch.shape[0]
    else:
        bs = None

    return bs
