# My imports
# None of my imports used

# Regular imports
from copy import deepcopy
from scipy import signal

import numpy as np
import mir_eval
import librosa
import random
import torch


# TODO - torch Tensor compatibility
# TODO - description of data formats in headings
# TODO - functions to convert all types of note formats to/from hz/midi


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


##################################################
# TO NOTES                                       #
##################################################


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

    # TODO - validation check?

    return pitches, intervals


##################################################
# TO STACKED NOTES                               #
##################################################


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

    # TODO - validation check?

    return stacked_notes


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

    # TODO - validation check?

    return times, pitch_list


def multi_pitch_to_pitch_list():
    """
    TODO
    active_pitches = []

    for i in range(pianoroll.shape[-1]): # For each frame
        # Determine the activations across this frame
        active_pitches += [librosa.midi_to_hz(np.where(pianoroll[:, i] != 0)[0] + lowest)]
    """
    pass


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

    # TODO - validation check?

    return stacked_pitch_list


def stacked_multi_pitch_to_stacked_pitch_list():
    pass


##################################################
# TO MULTI PITCH                                 #
##################################################


def notes_to_multi_pitch(pitches, intervals, times, profile):
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

    # Convert the pitches to number of semitones from lowest note
    pitches = np.round(pitches - profile.low).astype('uint')

    # Duplicate the array of times for each note and stack along a new axis
    times = np.concatenate([[times]] * max(1, len(pitches)), axis=0)

    # Determine the frame where each note begins and ends
    onsets = np.argmin((times <= intervals[..., :1]), axis=1) - 1
    offsets = np.argmin((times < intervals[..., 1:]), axis=1) - 1

    # Clip all offsets at last frame - they will end up at -1 from
    # previous operation if they occurred beyond last frame time
    offsets[offsets == -1] = num_frames - 1

    # Loop through each note
    for i in range(len(pitches)):
        # Populate the multi pitch array with activations for the note
        multi_pitch[pitches[i], onsets[i] : offsets[i] + 1] = 1

    # TODO - validation check?

    return multi_pitch


def pitch_list_to_multi_pitch(times, pitch_list, profile, tolerance=0.5):
    """
    Convert a MIDI pitch list into a dictionary of stacked pitch lists.

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

    # TODO - validation check?

    return multi_pitch


def stacked_multi_pitch_to_multi_pitch(stacked_multi_pitch):
    """
    Collapse stacked multi pitch arrays into a single representation.

    Parameters
    ----------
    stacked_multi_pitch : ndarray (S x F x T)
      Array of multiple discrete pitch activation maps
      S - number of slices in stack
      F - number of discrete pitches
      T - number of frames

    Returns
    ----------
    multi_pitch : ndarray (F x T)
      Discrete pitch activation map
      F - number of discrete pitches
      T - number of frames
    """

    # Collapse the stacked arrays into one using the max operation
    multi_pitch = np.max(stacked_multi_pitch, axis=-3)

    # TODO - validation check?

    return multi_pitch


##################################################
# TO STACKED MULTI PITCH                         #
##################################################


def stacked_notes_to_stacked_multi_pitch(stacked_notes, times, profile):
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
        stacked_multi_pitch.append(multi_pitch_to_stacked_multi_pitch(multi_pitch))

    # Collapse the list into an array
    stacked_multi_pitch = np.concatenate(stacked_multi_pitch)

    # TODO - validation check?

    return stacked_multi_pitch


def stacked_pitch_list_to_stacked_multi_pitch(stacked_pitch_list, profile):
    """
    Convert a stacked MIDI pitch list into a stack of multi pitch arrays.

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
        stacked_multi_pitch.append(multi_pitch_to_stacked_multi_pitch(multi_pitch))

    # Collapse the list into an array
    stacked_multi_pitch = np.concatenate(stacked_multi_pitch)

    # TODO - validation check?

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
    stacked_multi_pitch = np.expand_dims(multi_pitch, axis=0)

    # TODO - validation check?

    return stacked_multi_pitch


def tablature_to_stacked_multi_pitch():
    """
    TODO
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
    """
    pass


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
    profile : InstrumentProfile (instrument.py)
      Instrument profile detailing experimental setup

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

    # TODO - validation check?

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
    profile : InstrumentProfile (instrument.py)
      Instrument profile detailing experimental setup

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

        # Determine which frames have no note activations
        silent_frames = np.sum(multi_pitch, axis=0) == 0

        # Lower and upper pitch boundary for this degree of freedom
        lower_bound = tuning[dof] - profile.low
        upper_bound = lower_bound + profile.num_frets + 1

        # Bound the multi pitch array by the support of the degree of freedom
        multi_pitch = multi_pitch[lower_bound : upper_bound]

        # Determine which class has the highest activation across each frame
        highest_class = np.argmax(multi_pitch, axis=0)

        # Overwrite the highest class for the silent frames
        highest_class[silent_frames] = -1

        # Add the class membership to the tablature
        tablature += [np.expand_dims(highest_class, axis=0)]

    # Collapse the list to get the final tablature
    tablature = np.concatenate(tablature)

    # TODO - validation check?

    return tablature


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
    ambiguity : float or None (optional
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

    # TODO - validation check?

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

    # TODO - validation check?

    return onsets


##################################################
# TO STACKED ONSETS                              #
##################################################


def stacked_multi_pitch_to_stacked_onsets():
    pass


def stacked_notes_to_stacked_onsets():
    pass


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
    ambiguity : float or None (optional
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

    # Treat the time after offset as a note
    onset_times = np.copy(offset_times)

    if ambiguity is not None:
        # Add the ambiguity to the "note" duration
        offset_times += ambiguity
    else:
        # Make the duration zero
        offset_times = np.copy(onset_times)

    # Construct the intervals of the "note" following the offset
    post_note_intervals = np.concatenate((onset_times, offset_times), axis=-1)

    # Obtain the offsets using the note to multi pitch conversion
    offsets = notes_to_multi_pitch(pitches, post_note_intervals, times, profile)

    # TODO - validation check?

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

    # TODO - validation check?

    return offsets


##################################################
# TO STACKED OFFSETS                             #
##################################################


def stacked_multi_pitch_to_stacked_offsets():
    pass


def stacked_notes_to_stacked_offsets():
    pass


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

    # TODO - validation check?

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

    # TODO - validation check?

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

    # TODO - validation check?

    return times, pitch_list


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

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


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


# TODO - major cleanup needed for all of these functions


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


# TODO - should I be using a yield function instead of for batch in loader?


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
