# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from . import tools

# Regular imports
from abc import abstractmethod
from copy import deepcopy

import numpy as np
import os

__all__ = [
    'filter_notes_by_duration',
    'ComboEstimator',
    'Estimator',
    'MultiPitchWrapper',
    'StackedNoteTranscriber',
    'IterativeStackedNoteTranscriber',
    'NoteTranscriber',
    'IterativeNoteTranscriber',
    'StackedMultiPitchRefiner',
    'MultiPitchRefiner',
    'StackedPitchListWrapper',
    'PitchListWrapper',
    'TablatureWrapper',
    'Collapser',
    'StackedMultiPitchCollapser',
    'StackedNotesCollapser',
    'StackedPitchListCollapser',
    'StackedOnsetsWrapper',
    'StackedOffsetsWrapper'
]

# TODO - KeyChanger to simply change the dictionary key of an estimate??
# TODO - Copier to simply copy an entry under a different key??


def filter_notes_by_duration(pitches, intervals, threshold=0.):
    """
    Remove notes from a collection which have a duration less than a threshold
    TODO - add to tools and make same function for batched notes to use in here

    Parameters
    ----------
    pitches : ndarray (N)
      Array of pitches corresponding to notes
      N - number of notes
    intervals : ndarray (N x 2)
      Array of onset-offset time pairs corresponding to notes
      N - number of notes
    threshold : float
      Minimum duration (seconds) to keep a note - if set to zero, notes must have non-zero duration

    Returns
    ----------
    pitches : ndarray (N)
      Array of pitches corresponding to notes
      N - number of notes
    intervals : ndarray (N x 2)
      Array of onset-offset time pairs corresponding to notes
      N - number of notes
    """

    # Convert to batched notes for easy indexing
    batched_notes = tools.notes_to_batched_notes(pitches, intervals)
    # Calculate the duration of each note
    durations = batched_notes[:, 1] - batched_notes[:, 0]

    if threshold:
        # Remove notes with duration below the threshold
        batched_notes = batched_notes[durations >= threshold]
    else:
        # Remove zero-duration notes
        batched_notes = batched_notes[durations > threshold]

    # Convert back to loose note groups
    pitches, intervals = tools.batched_notes_to_notes(batched_notes)

    return pitches, intervals


##################################################
# ESTIMATORS                                     #
##################################################


class ComboEstimator(object):
    """
    A simple wrapper to run multiple estimators in succession.
    Order matters. For instance, a MultiPitchRefiner could be
    chained before a PitchListWrapper to use the refined
    predictions when generating pitch list estimations.
    """

    def __init__(self, estimators):
        """
        Initialize estimators and instantiate.

        Parameters
        ----------
        estimators : list of Estimator
          Estimators to use (in-order) when processing a track
        """

        self.estimators = estimators

    def process_track(self, raw_output, track=None):
        """
        Process the track independently using each estimator.

        Parameters
        ----------
        raw_output : dict
          Dictionary containing raw output relevant to estimation
        track : string or None (optional)
          Name of the track to use when writing estimates
        """

        # Copy the raw output dictionary and use it to hold estimates
        output = deepcopy(raw_output)

        # Loop through all of the estimators
        for estimator in self.estimators:
            # Process the track with the estimator and update the estimate dictionary
            output.update(estimator.process_track(output, track))

        return output

    def set_save_dirs(self, save_dir, sub_dirs=None):
        """
        Update the save directories for all of the estimators.

        Parameters
        ----------
        save_dir : string
          Directory under which to write output
        sub_dirs : list of string or None (optional)
          Sub-directories to use underneath 'save_dir' for each estimator
          Specifying None for an individual sub-directory
          will disable saving for the respective estimator
        """

        # Loop through all of the estimators
        for i, estimator in enumerate(self.estimators):
            if sub_dirs is None:
                # Do not add a sub-directory to the path
                new_dir = save_dir
            elif sub_dirs[i] is None:
                # Disable saving for the estimator
                new_dir = None
            else:
                # Append the specified sub-directory if it exists
                new_dir = os.path.join(save_dir, sub_dirs[i])

            # Update the save directory
            estimator.set_save_dir(new_dir)

    def reset_state(self):
        """
        Reset all the estimators in the combo.
        """

        # Loop through all of the estimators
        for estimator in self.estimators:
            # Reset the state of each
            estimator.reset_state()


class Estimator(object):
    """
    Implements a generic music information retrieval estimator.
    """

    def __init__(self, profile, estimates_key=None, save_dir=None):
        """
        Initialize parameters common to all estimators and instantiate.

        Parameters
        ----------
        profile : InstrumentProfile (instrument.py)
          Instrument profile detailing experimental setup
        estimates_key : string or None (optional)
          Key to use when packing estimates
        save_dir : string or None (optional)
          Directory where estimates for each track will be written
        """

        self.profile = profile

        # Set up the key to use for organizing estimates
        self.estimates_key = self.get_default_key() if estimates_key is None else estimates_key

        self.save_dir = None
        self.set_save_dir(save_dir)

    def set_save_dir(self, save_dir):
        """
        Simple helper function to set and create a new save directory.

        Parameters
        ----------
        save_dir : string or None (optional)
          Directory where estimates for each track will be written
        """

        self.save_dir = save_dir

        if self.save_dir is not None:
            # Create the specified directory if it does not already exist
            os.makedirs(self.save_dir, exist_ok=True)

    @staticmethod
    @abstractmethod
    def get_default_key():
        """
        Default key describing estimates in the event no key was provided.
        """

        return NotImplementedError

    @abstractmethod
    def pre_proc(self, raw_output):
        """
        This method can be overridden in order to insert extra steps.

        Parameters
        ----------
        raw_output : dict
          Dictionary containing raw output relevant to estimation

        Returns
        ----------
        raw_output : dict
          Copy of parameterized raw output
        """

        # Create a local copy of the output so it is only modified within scope
        raw_output = deepcopy(raw_output)

        return raw_output

    @abstractmethod
    def estimate(self, raw_output):
        """
        Obtain the estimate from the raw output.

        Parameters
        ----------
        raw_output : dict
          Dictionary containing raw output relevant to estimation
        """

        return NotImplementedError

    @abstractmethod
    def write(self, estimate, track):
        """
        Specify the protocol for writing the estimates.

        Parameters
        ----------
        estimate : object
          Estimate for a track
        track : string
          Name of the track being processed
        """

        return NotImplementedError

    @abstractmethod
    def reset_state(self):
        """
        Specify the protocol for resetting the state of the estimator.
        """

        pass

    def process_track(self, raw_output, track=None):
        """
        Combines pre_proc(), estimate(), and write(), and returns output in a dictionary.

        Parameters
        ----------
        raw_output : dict
          Dictionary containing raw output relevant to estimation
        track : string or None (optional)
          Name of the track being processed

        Returns
        ----------
        output : dict
          Estimate packed in a dictionary
        """

        # Perform any pre-processing steps
        raw_output = self.pre_proc(raw_output)
        # Obtain estimates for the track
        estimate = self.estimate(raw_output)

        if self.save_dir is not None:
            # Write the results to a text file
            self.write(estimate, track)

        # Return the output in a dictionary
        output = {self.estimates_key : estimate}

        return output


class MultiPitchWrapper(Estimator):
    """
    Abstracts several functions that are common to estimators dealing with multi pitch arrays.

    TODO - create wrapper for pitch list / note estimators as well?
           might not help too much to due stacked variations...
    """

    @staticmethod
    def get_default_key():
        """
        Default key for multi pitch activations.
        """

        return tools.KEY_MULTIPITCH

    def estimate(self, raw_output):
        """
        Here, just simply pass through the raw multi pitch data.

        Parameters
        ----------
        raw_output : dict
          Dictionary containing multi pitch

        Returns
        ----------
        multi_pitch : ndarray (... x F x T)
          Raw discrete pitch activation map
          F - number of discrete pitches
          T - number of frames
        """

        # Obtain the multi pitch activation map
        multi_pitch = tools.unpack_dict(raw_output, self.estimates_key)

        return multi_pitch

    def write(self, multi_pitch, track):
        """
        Write the multi pitch activation maps to a file.

        Parameters
        ----------
        multi_pitch : ndarray (... x F x T)
          Discrete pitch activation maps
          F - number of discrete pitches
          T - number of frames
        track : string
          Name of the track being processed
        """

        # Determine how to name the results
        tag = tools.get_tag(track)

        # Construct a path for saving the estimates
        path = os.path.join(self.save_dir, f'{tag}')

        # Save the multi pitch array
        np.save(path, multi_pitch)


class StackedNoteTranscriber(Estimator):
    """
    Estimate stacked notes from stacked multi pitch activation maps.
    """

    def __init__(self, profile, inhibition_window=None, minimum_duration=None,
                 multi_pitch_key=None, onsets_key=None, offsets_key=None,
                 estimates_key=None, save_dir=None):
        """
        Initialize parameters for the estimator.

        Parameters
        ----------
        See Estimator class for others...

        inhibition_window : float or None (optional)
          Amount of time after which another note of the same pitch cannot begin
        minimum_duration : float or None (optional)
          Minimum necessary duration to keep a note
        multi_pitch_key : string or None (optional)
          Key to use when unpacking multi pitch data
        onsets_key : string or None (optional)
          Key to use when unpacking onsets data
        offsets_key : string or None (optional)
          Key to use when unpacking offsets data
        """

        super().__init__(profile=profile,
                         estimates_key=estimates_key,
                         save_dir=save_dir)

        self.inhibition_window = inhibition_window
        self.minimum_duration = minimum_duration

        # Default the keys for unpacking relevant data
        self.multi_pitch_key = tools.KEY_MULTIPITCH if multi_pitch_key is None else multi_pitch_key
        self.onsets_key = tools.KEY_ONSETS if onsets_key is None else onsets_key
        self.offsets_key = tools.KEY_OFFSETS if offsets_key is None else offsets_key

    @staticmethod
    def get_default_key():
        """
        Default key for note estimates.
        """

        return tools.KEY_NOTES

    def estimate(self, raw_output):
        """
        Estimate notes for each slice of a stacked multi pitch activation map.

        Parameters
        ----------
        raw_output : dict
          Dictionary containing raw output relevant to estimation

        Returns
        ----------
        stacked_notes : dict
          Dictionary containing (slice -> (pitches, intervals)) pairs
        """

        # Obtain the multi pitch activation maps to transcribe
        stacked_multi_pitch = tools.unpack_dict(raw_output, self.multi_pitch_key)

        # Determine the number of slices in the stacked multi pitch array
        stack_size = stacked_multi_pitch.shape[-3]

        # Obtain the frame times associated with the activation maps
        times = tools.unpack_dict(raw_output, tools.KEY_TIMES)

        # Obtain the onsets and offsets from the raw output if they exist
        stacked_onsets = tools.unpack_dict(raw_output, self.onsets_key)
        stacked_offsets = tools.unpack_dict(raw_output, self.offsets_key)

        # If no onsets were provided, prepare a list of None's
        if stacked_onsets is None:
            stacked_onsets = [None] * stack_size

        # If no offsets were provided, prepare a list of None's
        if stacked_offsets is None:
            stacked_offsets = [None] * stack_size

        # Initialize a dictionary to hold the notes
        stacked_notes = dict()

        # Loop through the slices of the stack
        for slc in range(stack_size):
            # Obtain all of the transcription information for this slice
            multi_pitch, onsets, offsets = stacked_multi_pitch[slc], stacked_onsets[slc], stacked_offsets[slc]

            if self.inhibition_window is not None:
                if onsets is None:
                    # Default the onsets if they were not provided
                    onsets = tools.multi_pitch_to_onsets(multi_pitch)
                    # Remove trailing onsets within inhibition window of a previous onset
                    onsets = tools.inhibit_activations(onsets, times, self.inhibition_window)

            # Transcribe this slice of activations
            pitches, intervals = tools.multi_pitch_to_notes(multi_pitch, times, self.profile, onsets, offsets)

            if self.minimum_duration is not None:
                # Filter the notes by duration
                pitches, intervals = filter_notes_by_duration(pitches, intervals, self.minimum_duration)

            # Add the pitch-interval pairs to the stacked notes dictionary under the slice key
            stacked_notes.update(tools.notes_to_stacked_notes(pitches, intervals, slc))

        return stacked_notes

    def write(self, stacked_notes, track):
        """
        Write slice-wise note estimates to respective files.

        Parameters
        ----------
        stacked_notes : dict
          Dictionary containing (slice -> (pitches, intervals)) pairs
        track : string
          Name of the track being processed
        """

        # Obtain a list of the stacked note keys
        keys = list(stacked_notes.keys())

        # Determine how to name the results
        tag = tools.get_tag(track)

        # Loop through the slices of the stack
        for key in keys:
            # Add another tag for the degree of freedom if more than one
            slice_tag = f'{tag}_{key}' if len(stacked_notes) > 1 else f'{tag}'

            # Construct a path for saving the estimates
            path = os.path.join(self.save_dir, f'{slice_tag}.{tools.TXT_EXT}')

            # Extract the loose note groups from the stack
            pitches, intervals = stacked_notes[key]

            # Write the notes to the path
            tools.write_notes(pitches, intervals, path)


class IterativeStackedNoteTranscriber(StackedNoteTranscriber):
    """
    Estimate stacked notes from stacked multi pitch activation maps, one frame at a time.
    """

    def __init__(self, profile, inhibition_window=None, minimum_duration=None,
                 multi_pitch_key=None, onsets_key=None, offsets_key=None,
                 estimates_key=None, save_dir=None):
        """
        Initialize parameters for the estimator.

        Parameters
        ----------
        See StackedNoteTranscriber class...
        """

        super().__init__(profile=profile,
                         inhibition_window=inhibition_window,
                         minimum_duration=minimum_duration,
                         multi_pitch_key=multi_pitch_key,
                         onsets_key=onsets_key,
                         offsets_key=offsets_key,
                         estimates_key=estimates_key,
                         save_dir=save_dir)

        # Create an array to keep track of the previous pitch activations
        self.previous_activations = None
        # Create an array to keep track of onset times for active pitches
        self.active_pitches = None

        self.reset_state()

    def reset_state(self):
        """
        Zero-out the tracked state.
        """

        # Assume that no notes were previously active
        self.previous_activations = np.zeros((self.profile.get_num_dofs(),
                                              self.profile.get_range_len(), 1))

        # Set all onset times to zero
        self.active_pitches = np.zeros(self.previous_activations.shape)

    def estimate(self, raw_output):
        """
        Track notes for each slice of a stacked multi pitch activation map,
        acquiring note estimates when the notes are complete.

        TODO - verify transcription results are the same
        TODO - how to deal with active pitches when last frame is given

        Parameters
        ----------
        raw_output : dict
          Dictionary containing raw output relevant to estimation

        Returns
        ----------
        stacked_notes : dict
          Dictionary containing (slice -> (pitches, intervals)) pairs
        """

        # Obtain the multi pitch activation maps to transcribe
        stacked_multi_pitch = tools.unpack_dict(raw_output, self.multi_pitch_key)

        # Determine the number of slices in the stacked multi pitch array
        stack_size = stacked_multi_pitch.shape[-3]

        # Obtain the frame time associated with the activation maps
        time = tools.unpack_dict(raw_output, tools.KEY_TIMES)[-1:].item()

        # Obtain the onsets and offsets from the raw output if they exist
        stacked_onsets = tools.unpack_dict(raw_output, self.onsets_key)
        stacked_offsets = tools.unpack_dict(raw_output, self.offsets_key)

        # Append the new pitch activations to the last frame of activations
        activations = np.concatenate((self.previous_activations, stacked_multi_pitch), axis=-1)

        # If no onsets were provided, obtain them from the pitch activations
        if stacked_onsets is None:
            stacked_onsets = tools.stacked_multi_pitch_to_stacked_onsets(activations)[..., -1:]

        # If no offsets were provided, obtain them from the pitch activations
        if stacked_offsets is None:
            stacked_offsets = tools.stacked_multi_pitch_to_stacked_offsets(activations)[..., :-1]

        # Consider onsets where pitch is already active as offsets
        stacked_offsets = np.logical_or(stacked_offsets,
                                        np.logical_and(stacked_onsets, self.active_pitches)).astype(tools.FLOAT)

        # Initialize a dictionary to hold the notes
        stacked_notes = dict()

        # Loop through the slices of the stack
        for slc in range(stack_size):
            # Obtain the transcription information for this slice
            onsets, offsets = stacked_onsets[slc], stacked_offsets[slc]

            if self.inhibition_window is not None:
                # TODO
                pass

            # Obtain the indices of pitches which are no longer active
            offsets = offsets.squeeze() == 1
            # Determine the pitches which are no longer active
            pitches = self.profile.get_midi_range()[offsets]
            # Obtain the note onsets of these pitches
            intervals = self.active_pitches[slc, offsets]
            # Append the current time to get the note interval
            intervals = np.concatenate((intervals, time * np.ones(intervals.shape)), axis=-1)

            if self.minimum_duration is not None:
                # Filter the notes by duration
                pitches, intervals = filter_notes_by_duration(pitches, intervals, self.minimum_duration)

            # Convert to batched notes
            batched_notes = tools.notes_to_batched_notes(pitches, intervals)

            # Add the notes to the stacked notes dictionary under the slice key
            # TODO - is it absolutely necessary to transpose these???
            stacked_notes.update(tools.batched_notes_to_stacked_notes(batched_notes, True, slc))

        # Clear the onset times
        self.active_pitches[stacked_offsets == 1] = 0.
        # Add current time as onset of newly active pitches
        self.active_pitches[stacked_onsets == 1] = time

        # Update the previous activations
        self.previous_activations = stacked_multi_pitch

        return stacked_notes

    def get_active_stacked_multi_pitch(self):
        """
        Obtain the estimator's active notes as a stacked multi pitch array.

        Returns
        ----------
        stacked_multi_pitch : ndarray (S x F x T)
          Array of multiple discrete pitch activation maps
          S - number of slices in stack
          F - number of discrete pitches
          T - number of frames
        """

        # Create an array of zeros with the same size as the active pitch array
        stacked_multi_pitch = np.zeros(self.active_pitches.shape)
        # Change all nonzero values to one
        stacked_multi_pitch[self.active_pitches != 0] = 1

        return stacked_multi_pitch

    def get_active_stacked_notes(self, current_time=None):
        """
        Obtain the estimator's active notes as stacked notes.

        Parameters
        ----------
        current_time : float or None (Optional)
          The current tracked time, for adding duration to the note estimates

        Returns
        ----------
        stacked_notes : dict
          Dictionary containing (slice -> (pitches, intervals)) pairs
        """

        # Initialize a dictionary to hold the notes
        stacked_notes = dict()

        # Obtain the onset times of the current active pitches per slice
        active_pitch_onsets = self.active_pitches.squeeze(-1)

        # Determine the number of slices
        stack_size = active_pitch_onsets.shape[0]

        # Loop through the slices of the stack
        for slc in range(stack_size):
            # Acquire the active MIDI pitches
            pitches = self.profile.get_midi_range()[active_pitch_onsets[slc] != 0]
            # Acquire the onset times corresponding to the pitches
            onset_times = active_pitch_onsets[slc, active_pitch_onsets[slc] != 0]

            # Check if a time was provided
            if current_time is None:
                # Make the offsets the same as the onsets
                offset_times = onset_times
            else:
                # Make the offsets the provided time
                offset_times = current_time * np.ones(onset_times.shape)

            # Construct the note intervals
            intervals = np.concatenate(([onset_times], [offset_times]), axis=-1)

            # Add the notes to the stacked notes dictionary
            stacked_notes[slc] = pitches, intervals

        return stacked_notes


class NoteTranscriber(StackedNoteTranscriber):
    """
    Estimate notes from a multi pitch activation map.
    """

    def estimate(self, raw_output):
        """
        Estimate notes from a multi pitch activation map.

        Parameters
        ----------
        raw_output : dict
          Dictionary containing raw output relevant to estimation

        Returns
        ----------
        batched_notes : ndarray (N x 3)
          Array of note intervals and pitches by row
          N - number of notes
        """

        # Obtain the multi pitch activation map to transcribe
        multi_pitch = tools.unpack_dict(raw_output, self.multi_pitch_key)

        # Convert the multi pitch array to a stacked multi pitch array
        raw_output[self.multi_pitch_key] = tools.multi_pitch_to_stacked_multi_pitch(multi_pitch)

        # Obtain onsets and offsets from output if they exist
        onsets = tools.unpack_dict(raw_output, self.onsets_key)
        offsets = tools.unpack_dict(raw_output, self.offsets_key)

        if onsets is not None:
            # Convert onsets to a stacked onset activation map
            raw_output[self.onsets_key] = tools.multi_pitch_to_stacked_multi_pitch(onsets)

        if offsets is not None:
            # Convert offsets to a stacked offset activation map
            raw_output[self.offsets_key] = tools.multi_pitch_to_stacked_multi_pitch(offsets)

        # Call the parent class estimate function. Multi pitch is just a special
        # case of stacked multi pitch, where there is only one degree of freedom
        output = super().estimate(raw_output)

        # Add the estimated output to the raw output
        batched_notes = tools.notes_to_batched_notes(*tools.stacked_notes_to_notes(output))

        return batched_notes

    def write(self, batched_notes, track):
        """
        Write note estimates to a file.

        Parameters
        ----------
        batched_notes : ndarray (N x 3)
          Array of note intervals and pitches by row
          N - number of notes
        track : string
          Name of the track being processed
        """

        # Convert the batched notes to loose note groups
        pitches, intervals = tools.batched_notes_to_notes(batched_notes)

        # Stack the loose note groups
        stacked_notes = tools.notes_to_stacked_notes(pitches, intervals)

        # Call the parent function
        super().write(stacked_notes, track)


class IterativeNoteTranscriber(IterativeStackedNoteTranscriber):
    """
    Estimate notes from a multi pitch activation map, one frame at a time.
    """

    def reset_state(self):
        """
        Zero-out the tracked state.
        """

        # Assume that no notes were previously active
        self.previous_activations = np.zeros((1, self.profile.get_range_len(), 1))

        # Set all onset times to zero
        self.active_pitches = np.zeros(self.previous_activations.shape)

    def estimate(self, raw_output):
        """
        Track notes within a multi pitch activation map,
        acquiring note estimates when the notes are complete.

        TODO - verify transcription results are the same
        TODO - how to deal with active pitches when last frame is given

        Parameters
        ----------
        raw_output : dict
          Dictionary containing raw output relevant to estimation

        Returns
        ----------
        stacked_notes : dict
          Dictionary containing batched notes
        """

        # Obtain the multi pitch activation map to transcribe
        multi_pitch = tools.unpack_dict(raw_output, self.multi_pitch_key)

        # Convert the multi pitch array to a stacked multi pitch array
        raw_output[self.multi_pitch_key] = tools.multi_pitch_to_stacked_multi_pitch(multi_pitch)

        # Obtain onsets and offsets from output if they exist
        onsets = tools.unpack_dict(raw_output, self.onsets_key)
        offsets = tools.unpack_dict(raw_output, self.offsets_key)

        if onsets is not None:
            # Convert onsets to a stacked onset activation map
            raw_output[self.onsets_key] = tools.multi_pitch_to_stacked_multi_pitch(onsets)

        if offsets is not None:
            # Convert offsets to a stacked offset activation map
            raw_output[self.offsets_key] = tools.multi_pitch_to_stacked_multi_pitch(offsets)

        # Call the parent class estimate function. Multi pitch is just a special
        # case of stacked multi pitch, where there is only one degree of freedom
        stacked_notes = super().estimate(raw_output)

        # Unpack the batched notes
        batched_notes = tools.stacked_notes_to_batched_notes(stacked_notes, True)

        return batched_notes


class StackedMultiPitchRefiner(MultiPitchWrapper):
    """
    Refine stacked multi pitch activation maps, after using them to make note
    predictions, by converting note estimates back into multi pitch activation.
    """

    def __init__(self, profile, notes_key=None, estimates_key=None, save_dir=None):
        """
        Initialize parameters for the estimator.

        Parameters
        ----------
        See Estimator class...

        notes_key : string or None (optional)
          Key to use when unpacking notes data
        """

        super().__init__(profile=profile,
                         estimates_key=estimates_key,
                         save_dir=save_dir)

        # Default the key for unpacking relevant data
        self.notes_key = tools.KEY_NOTES if notes_key is None else notes_key

    def estimate(self, raw_output):
        """
        Refine a stacked multi pitch activation map.

        Parameters
        ----------
        raw_output : dict
          Dictionary containing raw output relevant to estimation

        Returns
        ----------
        stacked_multi_pitch : ndarray (S x F x T)
          Array of multiple discrete pitch activation maps
          S - number of slices in stack
          F - number of discrete pitches
          T - number of frames
        """

        # Extract pre-existing stacked note estimates
        stacked_notes = tools.unpack_dict(raw_output, self.notes_key)

        # Convert the batched notes in each slice to loose note groups
        stacked_notes = tools.apply_func_stacked_representation(stacked_notes, tools.batched_notes_to_notes)

        # Obtain the frame times associated with the multi pitch array
        times = tools.unpack_dict(raw_output, tools.KEY_TIMES)

        # Convert the stacked notes back into stacked multi pitch activation maps
        stacked_multi_pitch = tools.stacked_notes_to_stacked_multi_pitch(stacked_notes, times, self.profile)

        return stacked_multi_pitch


class MultiPitchRefiner(StackedMultiPitchRefiner):
    """
    Refine a multi pitch activation map, after using it to make note
    predictions, by converting note estimates back into multi pitch activation.
    """

    def estimate(self, raw_output):
        """
        Refine a multi pitch activation map.

        Parameters
        ----------
        raw_output : dict
          Dictionary containing raw output relevant to estimation

        Returns
        ----------
        multi_pitch : ndarray (F x T)
          Discrete pitch activation map
          F - number of discrete pitches
          T - number of frames
        """

        # Extract pre-existing note estimates
        batched_notes = tools.unpack_dict(raw_output, self.notes_key)

        # Convert the batched notes to loose note groups
        pitches, intervals = tools.batched_notes_to_notes(batched_notes)

        # Obtain the frame times associated with the multi pitch array
        times = tools.unpack_dict(raw_output, tools.KEY_TIMES)

        # Convert the notes back into a multi pitch array
        multi_pitch = tools.notes_to_multi_pitch(pitches, intervals, times, self.profile)

        return multi_pitch


class StackedPitchListWrapper(Estimator):
    """
    Wrapper for converting stacked multi pitch activations to stacked pitch lists.
    """

    def __init__(self, profile, multi_pitch_key=None, estimates_key=None, save_dir=None):
        """
        Initialize parameters for the estimator.

        Parameters
        ----------
        See Estimator class...

        multi_pitch_key : string or None (optional)
          Key to use when unpacking multi pitch data
        """

        super().__init__(profile=profile,
                         estimates_key=estimates_key,
                         save_dir=save_dir)

        # Default the key for unpacking relevant data
        self.multi_pitch_key = tools.KEY_MULTIPITCH if multi_pitch_key is None else multi_pitch_key

    @staticmethod
    def get_default_key():
        """
        Default key for pitch lists.
        """

        return tools.KEY_PITCHLIST

    def estimate(self, raw_output):
        """
        Convert stacked multi pitch activations to stacked pitch lists.

        Parameters
        ----------
        raw_output : dict
          Dictionary containing raw output relevant to estimation

        Returns
        ----------
        stacked_pitch_list : dict
          Dictionary containing (slice -> (times, pitch_list)) pairs
        """

        # Obtain the stacked multi pitch activation maps
        stacked_multi_pitch = tools.unpack_dict(raw_output, self.multi_pitch_key)

        # Obtain the frame times associated with the stacked activation map
        times = tools.unpack_dict(raw_output, tools.KEY_TIMES)

        # Perform the conversion
        stacked_pitch_list = tools.stacked_multi_pitch_to_stacked_pitch_list(stacked_multi_pitch, times, self.profile)

        return stacked_pitch_list

    def write(self, stacked_pitch_list, track):
        """
        Write slice-wise pitch estimates to respective files.

        Parameters
        ----------
        stacked_pitch_list : dict
          Dictionary containing (slice -> (times, pitch_list)) pairs
        track : string
          Name of the track being processed
        """

        # Obtain a list of the stacked pitch list keys
        keys = list(stacked_pitch_list.keys())

        # Determine how to name the results
        tag = tools.get_tag(track)

        # Loop through the slices of the stack
        for key in keys:
            # Add another tag for the degree of freedom if more than one
            slice_tag = f'{tag}_{key}' if len(stacked_pitch_list) > 1 else f'{tag}'

            # Construct a path for saving the estimates
            path = os.path.join(self.save_dir, f'{slice_tag}.{tools.TXT_EXT}')

            # Extract the pitch list from the stack
            times, pitch_list = stacked_pitch_list[key]

            # Write the notes to the path
            tools.write_pitch_list(times, pitch_list, path)


class PitchListWrapper(StackedPitchListWrapper):
    """
    Wrapper for converting a multi pitch activation map to a pitch list.
    """

    def estimate(self, raw_output):
        """
        Convert a multi pitch activation map to a pitch list.

        Parameters
        ----------
        raw_output : dict
          Dictionary containing raw output relevant to estimation

        Returns
        ----------
        times : ndarray (N)
          Time in seconds of beginning of each frame
          N - number of time samples (frames)
        pitch_list : list of ndarray (N x [...])
          Array of pitches corresponding to notes
          N - number of pitch observations (frames)
        """

        # Obtain the multi pitch activation map
        multi_pitch = tools.unpack_dict(raw_output, self.multi_pitch_key)

        # Obtain the frame times associated with the activation map
        times = tools.unpack_dict(raw_output, tools.KEY_TIMES)

        # Perform the conversion
        pitch_list = tools.multi_pitch_to_pitch_list(multi_pitch, self.profile)

        return times, pitch_list

    def write(self, pitch_list, track):
        """
        Write pitch estimates to a file.

        Parameters
        ----------
        pitch_list : tuple containing
          times : ndarray (N)
            Time in seconds of beginning of each frame
            N - number of time samples (frames)
          pitch_list : list of ndarray (N x [...])
            Array of pitches corresponding to notes
            N - number of pitch observations (frames)
        track : string
          Name of the track being processed
        """

        # Stack the pitch list
        stacked_pitch_list = tools.pitch_list_to_stacked_pitch_list(*pitch_list)

        # Call the parent function
        super().write(stacked_pitch_list, track)


class TablatureWrapper(MultiPitchWrapper):
    """
    Wrapper for converting tablature to multi pitch.
    """

    def __init__(self, profile, tablature_key=None, estimates_key=None, save_dir=None):
        """
        Initialize parameters for the estimator.

        Parameters
        ----------
        See Estimator class...

        tablature_key : string or None (optional)
          Key to use when unpacking tablature data
        """

        super().__init__(profile=profile,
                         estimates_key=estimates_key,
                         save_dir=save_dir)

        # Default the key for unpacking relevant data
        self.tablature_key = tools.KEY_TABLATURE if tablature_key is None else tablature_key

    def estimate(self, raw_output):
        """
        Convert tablature into a stacked multi pitch activation map.

        Parameters
        ----------
        raw_output : dict
          Dictionary containing raw output relevant to estimation

        Returns
        ----------
        multi_pitch : ndarray (S x F x T)
          Discrete pitch activation map
          S - number of slices in stack
          F - number of discrete pitches
          T - number of frames
        """

        # Obtain the tablature
        tablature = tools.unpack_dict(raw_output, self.tablature_key)

        # Perform the conversion
        multi_pitch = tools.tablature_to_stacked_multi_pitch(tablature, self.profile)

        return multi_pitch


class Collapser(Estimator):
    """
    Abstracts initialization functionality for wrappers meant to collapse stacked representations.
    """

    def __init__(self, profile, stacked_key=None, estimates_key=None, save_dir=None):
        """
        Initialize parameters for the estimator.

        Parameters
        ----------
        See Estimator class...

        stacked_key : string or None (optional)
          Key to use when unpacking stacked data
        """

        super().__init__(profile=profile,
                         estimates_key=estimates_key,
                         save_dir=save_dir)

        # Default the key for unpacking stacked data
        self.stacked_key = self.estimates_key if stacked_key is None else stacked_key


class StackedMultiPitchCollapser(Collapser, MultiPitchWrapper):
    """
    Wrapper for collapsing stacked multi pitch.
    """

    def estimate(self, raw_output):
        """
        Convert a stacked multi pitch activation map into a single multi pitch representation.

        Parameters
        ----------
        raw_output : dict
          Dictionary containing raw output relevant to estimation

        Returns
        ----------
        multi_pitch : ndarray (F x T)
          Discrete pitch activation map
          F - number of discrete pitches
          T - number of frames
        """

        # Obtain the multi pitch data
        stacked_multi_pitch = tools.unpack_dict(raw_output, self.stacked_key)

        # Perform the collapsing
        multi_pitch = tools.stacked_multi_pitch_to_multi_pitch(stacked_multi_pitch)

        return multi_pitch


class StackedNotesCollapser(Collapser, NoteTranscriber):
    """
    Wrapper for collapsing stacked notes.
    """

    def estimate(self, raw_output):
        """
        Convert stacked notes into a single notes representation.

        Parameters
        ----------
        raw_output : dict
          Dictionary containing raw output relevant to estimation

        Returns
        ----------
        batched_notes : ndarray (N x 3)
          Array of note intervals and pitches by row
          N - number of notes
        """

        # Obtain the stacked notes data
        stacked_notes = tools.unpack_dict(raw_output, self.stacked_key)

        # Collapse the stacked notes representation into a batched representation
        batched_notes = tools.notes_to_batched_notes(*tools.stacked_notes_to_notes(stacked_notes))

        return batched_notes


class StackedPitchListCollapser(Collapser, PitchListWrapper):
    """
    Wrapper for collapsing a stacked pitch list.
    """

    def estimate(self, raw_output):
        """
        Convert a stacked pitch list into a single representation.

        Parameters
        ----------
        raw_output : dict
          Dictionary containing raw output relevant to estimation

        Returns
        ----------
        times : ndarray (N)
          Time in seconds of beginning of each frame
          N - number of time samples (frames)
        pitch_list : list of ndarray (N x [...])
          Array of pitches corresponding to notes
          N - number of pitch observations (frames)
        """

        # Obtain the stacked pitch list data
        stacked_pitch_list = tools.unpack_dict(raw_output, self.stacked_key)

        # Collapse the stacked representation into a single pitch list
        times, pitch_list = tools.stacked_pitch_list_to_pitch_list(stacked_pitch_list)

        return times, pitch_list


class StackedOnsetsWrapper(MultiPitchWrapper):
    """
    Wrapper for obtaining stacked onsets activations from stacked multi pitch.
    """

    def __init__(self, profile, multi_pitch_key=None, estimates_key=None, save_dir=None):
        """
        Initialize parameters for the estimator.

        Parameters
        ----------
        See MultiPitchWrapper class...

        multi_pitch_key : string or None (optional)
          Key to use when unpacking multi pitch data
        """

        super().__init__(profile=profile,
                         estimates_key=estimates_key,
                         save_dir=save_dir)

        # Default the key for unpacking relevant data
        self.multi_pitch_key = tools.KEY_MULTIPITCH if multi_pitch_key is None else multi_pitch_key

    @staticmethod
    def get_default_key():
        """
        Default key for onsets activation maps.
        """

        return tools.KEY_ONSETS

    def estimate(self, raw_output):
        """
        Convert a stacked multi pitch activation map into a stacked onsets activation map.

        Parameters
        ----------
        raw_output : dict
          Dictionary containing raw output relevant to estimation

        Returns
        ----------
        stacked_onsets : ndarray (S x F x T)
          Array of multiple discrete onset activation maps
          S - number of slices in stack
          F - number of discrete pitches
          T - number of frames
        """

        # Obtain the multi pitch data
        stacked_multi_pitch = tools.unpack_dict(raw_output, self.multi_pitch_key)

        # Perform the conversion
        stacked_onsets = tools.stacked_multi_pitch_to_stacked_onsets(stacked_multi_pitch)

        return stacked_onsets


class StackedOffsetsWrapper(StackedOnsetsWrapper):
    """
    Wrapper for obtaining stacked offsets activations from stacked multi pitch.
    """

    @staticmethod
    def get_default_key():
        """
        Default key for offsets activation maps.
        """

        return tools.KEY_OFFSETS

    def estimate(self, raw_output):
        """
        Convert a stacked multi pitch activation map into a stacked offsets activation map.

        Parameters
        ----------
        raw_output : dict
          Dictionary containing raw output relevant to estimation

        Returns
        ----------
        stacked_offsets : ndarray (S x F x T)
          Array of multiple discrete offset activation maps
          S - number of slices in stack
          F - number of discrete pitches
          T - number of frames
        """

        # Obtain the multi pitch data
        stacked_multi_pitch = tools.unpack_dict(raw_output, self.multi_pitch_key)

        # Perform the conversion
        stacked_offsets = tools.stacked_multi_pitch_to_stacked_offsets(stacked_multi_pitch)

        return stacked_offsets
