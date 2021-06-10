# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from . import tools

# Regular imports
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.multipitch import evaluate as evaluate_frames
from abc import abstractmethod
from scipy.stats import hmean
from copy import deepcopy

import numpy as np
import sys
import os

EPSILON = sys.float_info.epsilon

# TODO - add warning when unpack returns None
# TODO - none of the stacked evaluators have been tested independently
#      - they will likely break during append, average, log, write, etc.

##################################################
# HELPER FUNCTIONS / RESULTS DICTIONARY          #
##################################################


def average_results(results):
    """
    Obtain the average across all tracked results for each metric
    in a results dictionary.

    Parameters
    ----------
    results : dictionary
      Dictionary containing results of tracks arranged by metric

    Returns
    ----------
    average : dictionary
      Dictionary with a single value for each metric
    """

    # Only modify a local copy which will be returned
    average = deepcopy(results)

    # Loop through the keys in the dictionary
    for key in average.keys():
        # Check if the entry is another dictionary
        if isinstance(average[key], dict):
            # Recursively call this function
            average[key] = average_results(average[key])
        else:
            # Check if the entry is a NumPy array or list - leave it alone otherwise
            if isinstance(average[key], np.ndarray) or isinstance(average[key], list):
                # Take the average of all entries and convert to float (necessary for logger)
                average[key] = float(np.mean(average[key]))

    return average


def append_results(tracked_results, new_results):
    """
    Combine two results dictionaries. This function is more general than
    the signature suggests.

    Parameters
    ----------
    tracked_results and new_results : dictionary
      Dictionaries containing results of tracks arranged by metric

    Returns
    ----------
    tracked_results : dictionary
      Dictionary with all results appended along the metric
    """

    # Only modify a local copy which will be returned
    tracked_results = deepcopy(tracked_results)

    # Loop through the keys in the new dictionary
    for key in new_results.keys():
        # Check if the key already exists in the current dictionary
        if key not in tracked_results.keys():
            # Add the untracked entry
            tracked_results[key] = new_results[key]
        # Check if the entry is another dictionary
        elif isinstance(new_results[key], dict):
            # Recursively call this function
            tracked_results[key] = append_results(tracked_results[key], new_results[key])
        else:
            # Append the new entry (or entries) to the current entry
            tracked_results[key] = np.append(tracked_results[key], new_results[key])

    return tracked_results


def log_results(results, writer, step=0, patterns=None, tag=''):
    """
    Log results using TensorBoardX.

    Parameters
    ----------
    results : dictionary
      Dictionary containing results of tracks arranged by metric
    writer : tensorboardX.SummaryWriter
      Writer object being used to log results
    step : int
      Current iteration in whatever process (e.g. training)
    patterns : list of string or None (optional)
      Only write metrics containing these patterns (e.g. ['f1', 'pr']) (None for all metrics)
    tag : string
      Tag for organizing different types of results (e.g. 'validation')
    """

    # Loop through the keys in the dictionary
    for key in results.keys():
        # Extract the next entry
        entry = results[key]

        # Check if the entry is another dictionary
        if isinstance(entry, dict):
            # Add the key to the tag and call this function recursively
            log_results(entry, writer, step, patterns, tag + f'/{key}')
        else:
            # Check if the key matches the specified patterns
            if pattern_match(key, patterns) or patterns is None:
                # Log the entry under the specified key
                writer.add_scalar(f'{tag}/{key}', entry, global_step=step)


def write_results(results, file, patterns=None, verbose=False):
    """
    Write result dictionary to a text file.

    Parameters
    ----------
    results : dictionary
      Dictionary containing results of tracks arranged by metric
    file : TextIOWrapper
      File open in write mode
    patterns : list of string or None (optional)
      Only write metrics containing these patterns (e.g. ['f1', 'pr']) (None for all metrics)
    verbose : bool
      Whether to print to console whatever is written to the file
    """

    # Loop through the keys in the dictionary
    for key in results.keys():
        # Check if the key's entry is another dictionary
        if isinstance(results[key], dict):
            # Write a header to the file
            tools.write_and_print(file, f'-----{key}-----', verbose, '\n')
            # Call this function recursively
            write_results(results[key], file, patterns, verbose)
            # Write an empty line
            tools.write_and_print(file, '', verbose, '\n')
        else:
            # Check if the key matches the specified patterns
            if pattern_match(key, patterns) or patterns is None:
                # Write the metric and corresponding result to the file
                tools.write_and_print(file, f' {key} : {results[key]}', verbose, '\n')

    # Write an empty line
    tools.write_and_print(file, '', verbose, '\n')


def pattern_match(query, patterns=None):
    """
    Simple helper function to see if a query matches a list of strings, even if partially.

    Parameters
    ----------
    query : string
      String to check for matches
    patterns : list of string or None (optional)
      Patterns to reference, return False if unspecified

    Returns
    ----------
    match : bool
      Whether the query matches some pattern, fully or partially
    """

    # Default the returned value
    match = False

    # Check if there are any patterns to analyze
    if patterns is not None:
        # Compare the query to each pattern
        match = any([p in query for p in patterns])

    return match


##################################################
# EVALUATORS                                     #
##################################################


class Evaluator(object):
    """
    Implements a generic music information retrieval evaluator.
    """

    def __init__(self, key, save_dir, patterns, verbose):
        """
        Initialize parameters common to all evaluators and instantiate.

        Parameters
        ----------
        key : string
          Key to use when unpacking data and organizing results
        save_dir : string or None (optional)
          Directory where results for each track will be written
        patterns : list of string or None (optional)
          Only write/log metrics containing these patterns (e.g. ['f1', 'pr']) (None for all metrics)
        verbose : bool
          Whether to print any written text to console as well
        """

        self.key = key

        self.save_dir = None
        self.set_save_dir(save_dir)

        self.patterns = None
        self.set_patterns(patterns)

        self.verbose = None
        self.set_verbose(verbose)

        # Initialize dictionary to track results
        self.results = None
        self.reset_results()

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

    def set_patterns(self, patterns):
        """
        Simple helper function to set new patterns.

        Parameters
        ----------
        patterns : list of string or None (optional)
          Only write/log metrics containing these patterns (e.g. ['f1', 'pr']) (None for all metrics)
        """

        self.patterns = patterns

    def set_verbose(self, verbose):
        """
        Simple helper function to set a new verbose flag.

        Parameters
        ----------
        verbose : bool
          Whether to print any written text to console as well
        """

        self.verbose = verbose

    def reset_results(self):
        """
        Reset tracked results to empty dictionary.
        """

        self.results = dict()

    def average_results(self):
        """
        Return the average of the currently tracked results.

        Returns
        ----------
        average : dictionary
          Dictionary with a single value for each metric
        """

        # Average the tracked results
        average = average_results(self.results)

        return average

    def get_key(self):
        """
        Obtain the key being used for the Evaluator.

        Returns
        ----------
        key : string
          Key to use when unpacking data and organizing results
        """

        if self.key is None:
            # Default the key
            key = self.get_default_key()
        else:
            # Use the provided key
            key = self.key

        return key

    @staticmethod
    @abstractmethod
    def get_default_key():
        """
        Provide the default key to use in the event no key was provided.
        """

        return NotImplementedError

    def unpack(self, data):
        """
        Unpack the relevant entry for evaluation if
        a dictionary is provided and the entry exists.

        Parameters
        ----------
        data : object
          Presumably either a dictionary containing ground-truth
          or model output, or the already-unpacked entry

        Returns
        ----------
        data : object
          Unpacked entry or same object provided if no dictionary
        """

        # Determine the relevant key for evaluation
        key = self.get_key()

        # Check if a dictionary was provided and if the key is in the dictionary
        data = tools.try_unpack_dict(data, key)

        return data

    def pre_proc(self, estimated, reference):
        """
        Handle both dictionary input as well as relevant input for
        both estimated and reference data.

        Note: This method can be overridden in order to insert extra steps.

        Parameters
        ----------
        estimated : object
          Dictionary containing ground-truth or the already-unpacked entry
        reference : object
          Dictionary containing model output or the already-unpacked entry

        Returns
        ----------
        estimated : object
          Estimate relevant to the evaluation
        reference : object
          Reference relevant to the evaluation
        """

        # Unpacked estimate and reference if dictionaries were provided
        estimated = self.unpack(estimated)
        reference = self.unpack(reference)

        return estimated, reference

    @abstractmethod
    def evaluate(self, estimated, reference):
        """
        Evaluate an estimate with respect to a reference.

        Parameters
        ----------
        estimated : object
          Estimate relevant to the evaluation or the dictionary containing it
        reference : object
          Reference relevant to the evaluation or the dictionary containing it
        """

        return NotImplementedError

    def write(self, results, track=None):
        """
        Write the results dictionary to a text file if a save directory was specified.

        Parameters
        ----------
        results : dictionary
          Dictionary containing results of tracks arranged by metric
        track : string
          Name of the track being processed
        """

        if self.save_dir is not None:
            # Determine how to name the results
            tag = tools.get_tag(track)

            if self.verbose:
                # Print the track name to console as a header to the results
                print(f'Evaluating track: {tag}')

            # Construct a path for the results
            results_path = os.path.join(self.save_dir, f'{tag}.{tools.TXT_EXT}')

            # Make sure all directories exist (there can be directories in the track name)
            os.makedirs(os.path.dirname(results_path), exist_ok=True)

            # Open a file at the path with writing permissions
            with open(results_path, 'w') as results_file:
                # Write the results to a text file
                write_results(results, results_file, self.patterns, self.verbose)

    def get_track_results(self, estimated, reference, track=None):
        """
        Calculate the results, write them, and track them within the evaluator.

        Parameters
        ----------
        estimated : object
          Estimate relevant to the evaluation or the dictionary containing it
        reference : object
          Reference relevant to the evaluation or the dictionary containing it
        track : string
          Name of the track being processed

        Returns
        ----------
        results : dictionary
          Dictionary containing results of tracks arranged by metric
        """

        # Make sure the estimated and reference data are unpacked
        estimated, reference = self.pre_proc(estimated, reference)

        # Calculate the results
        results = self.evaluate(estimated, reference)

        # Add the results to the tracked dictionary
        self.results = append_results(self.results, results)

        # Write the results
        self.write(results, track)

        return results

    def finalize(self, writer, step=0):
        """
        Log the averaged results using TensorBoardX and reset the results tracking.

        Parameters
        ----------
        writer : tensorboardX.SummaryWriter
          Writer object being used to log results
        step : int
          Current iteration in whatever process (e.g. training)
        """

        # Average the currently tracked results
        average = self.average_results()

        # Log the currently tracked results
        log_results(average, writer, step, patterns=self.patterns, tag=tools.VAL)

        # Reset the tracked results
        self.reset_results()


class ComboEvaluator(Evaluator):
    """
    Packages multiple evaluators into one modules.
    """

    def __init__(self, evaluators, save_dir=None, patterns=None, verbose=False):
        """
        Initialize parameters for the evaluator.

        Parameters
        ----------
        See Evaluator class...

        evaluators : list of Evaluator
          All of the evaluators to run
        """

        self.evaluators = evaluators

        super().__init__(None, save_dir, patterns, verbose)

    def reset_results(self):
        """
        Reset tracked results of each evaluator in the collection.
        """

        # Loop through the evaluators
        for evaluator in self.evaluators:
            # Reset the respective results dictionary so it is empty
            evaluator.reset_results()

    def average_results(self):
        """
        Return the average of the currently tracked results across all evaluators.

        Returns
        ----------
        average : dictionary
          Dictionary with results dictionary entries for each evaluator
        """

        # Initialize an empty dictionary for the average results
        average = dict()

        # Loop through the evaluators
        for evaluator in self.evaluators:
            # Average the tracked results for the evaluator
            # and place in average results under evaluator's key
            results = average_results(evaluator.results)

            # Check if there is already an entry for the evaluator's key
            if tools.query_dict(average, evaluator.get_key()):
                # Add new entries to the results
                average[evaluator.get_key()].update(results)
            else:
                # Create a new entry for the results
                average[evaluator.get_key()] = results

        return average

    @staticmethod
    @abstractmethod
    def get_default_key():
        """
        This should not be called directly on a ComboEvaluator.
        """

        return NotImplementedError

    @abstractmethod
    def evaluate(self, estimated, reference):
        """
        This should not be called directly on a ComboEvaluator.
        """

        return NotImplementedError

    def get_track_results(self, estimated, reference, track=None):
        """
        Very similar to parent method, except file is written after results are
        calculated for each evaluator and packaged into a single dictionary.

        Parameters
        ----------
        estimated : object
          Estimate relevant to the evaluation or the dictionary containing it
        reference : object
          Reference relevant to the evaluation or the dictionary containing it
        track : string
          Name of the track being processed

        Returns
        ----------
        results : dictionary
          Dictionary containing results of tracks arranged by metric
        """

        # Copy the raw output dictionary and use it to hold estimates
        results = {}

        # Loop through the evaluators
        for evaluator in self.evaluators:
            # Make sure the estimated and reference data are unpacked
            estimated_, reference_ = evaluator.pre_proc(estimated, reference)

            # Calculate the results
            new_results = evaluator.evaluate(estimated_, reference_)

            # Check if there is already an entry for the evaluator's key
            if tools.query_dict(results, evaluator.get_key()):
                # Add new entries to the results
                results[evaluator.get_key()].update(new_results)
            else:
                # Create a new entry for the results
                results[evaluator.get_key()] = new_results

            # Add the results to the tracked dictionary
            evaluator.results = append_results(evaluator.results, new_results)

        # Write the results
        self.write(results, track)

        return results


class LossWrapper(Evaluator):
    """
    Simple wrapper for tracking, writing, and logging loss.
    """

    def __init__(self, key=None, save_dir=None, patterns=None, verbose=False):
        """
        Initialize parameters for the evaluator.

        Parameters
        ----------
        See Evaluator class...
        """

        super().__init__(key, save_dir, patterns, verbose)

    @staticmethod
    def get_default_key():
        """
        Default key for loss.
        """

        return tools.KEY_LOSS

    def evaluate(self, estimated, reference=None):
        """
        Simply return loss in a new results dictionary.

        Parameters
        ----------
        estimated : ndarray
          Single loss value in a NumPy array
        reference : irrelevant

        Returns
        ----------
        results : dict
          Dictionary containing loss
        """

        # Package the results into a dictionary
        results = estimated

        return results


class StackedMultipitchEvaluator(Evaluator):
    """
    Implements an evaluator for stacked multi pitch activation maps, i.e.
    independent multi pitch estimations across degrees of freedom or instruments.
    """

    def __init__(self, key=None, save_dir=None, patterns=None, verbose=False):
        """
        Initialize parameters for the evaluator.

        Parameters
        ----------
        See Evaluator class...
        """

        super().__init__(key, save_dir, patterns, verbose)

    @staticmethod
    def get_default_key():
        """
        Default key for multi pitch activation maps.
        """

        return tools.KEY_MULTIPITCH

    def evaluate(self, estimated, reference):
        """
        Evaluate a stacked multi pitch estimate with respect to a reference.

        Parameters
        ----------
        estimated : ndarray (S x F x T)
          Array of multiple discrete pitch activation maps
          S - number of slices in stack
          F - number of discrete pitches
          T - number of frames
        reference : ndarray (S x F x T)
          Array of multiple discrete pitch activation maps
          Dimensions same as estimated

        Returns
        ----------
        results : dict
          Dictionary containing precision, recall, and f-measure
        """

        # Determine the shape necessary to flatten the last two dimensions
        flatten_shape = estimated.shape[:-2] + tuple([-1])

        # Flatten the estimated and reference data
        flattened_multi_pitch_est = np.reshape(estimated, flatten_shape)
        flattened_multi_pitch_ref = np.reshape(reference, flatten_shape)

        # Determine the number of correct predictions,
        # where estimated activation lines up with reference
        num_correct = np.sum(flattened_multi_pitch_est * flattened_multi_pitch_ref, axis=-1)

        # Count the number of activations predicted
        num_predicted = np.sum(flattened_multi_pitch_est, axis=-1)
        # Count the number of activations referenced
        num_ground_truth = np.sum(flattened_multi_pitch_ref, axis=-1)

        # Calculate precision and recall
        precision = num_correct / (num_predicted + EPSILON)
        recall = num_correct / (num_ground_truth + EPSILON)

        # Calculate the f1-score using the harmonic mean formula
        f_measure = hmean([precision + EPSILON, recall + EPSILON]) - EPSILON

        # Package the results into a dictionary
        results = {
            tools.KEY_PRECISION : precision,
            tools.KEY_RECALL : recall,
            tools.KEY_F1 : f_measure
        }

        return results


class MultipitchEvaluator(StackedMultipitchEvaluator):
    """
    Implements an evaluator for multi pitch activation maps.
    """

    def __init__(self, key=None, save_dir=None, patterns=None, verbose=False):
        """
        Initialize parameters for the evaluator.

        Parameters
        ----------
        See Evaluator class...
        """

        super().__init__(key, save_dir, patterns, verbose)

    def evaluate(self, estimated, reference):
        """
        Evaluate a multi pitch estimate with respect to a reference.

        Parameters
        ----------
        estimated : ndarray (F x T)
          Predicted discrete pitch activation map
          F - number of discrete pitches
          T - number of frames
        reference : ndarray (F x T)
          Ground-truth discrete pitch activation map
          Dimensions same as estimated

        Returns
        ----------
        results : dict
          Dictionary containing precision, recall, and f-measure
        """

        # Convert the multi pitch arrays to stacked multi pitch arrays
        stacked_multi_pitch_est = tools.multi_pitch_to_stacked_multi_pitch(estimated)
        stacked_multi_pitch_ref = tools.multi_pitch_to_stacked_multi_pitch(reference)

        # Call the parent class evaluate function. Multi pitch is just a special
        # case of stacked multi pitch, where there is only one degree of freedom
        results = super().evaluate(stacked_multi_pitch_est, stacked_multi_pitch_ref)

        # Average the results across the degree of freedom - i.e. collapse extraneous dimension
        results = average_results(results)

        return results


class StackedNoteEvaluator(Evaluator):
    """
    Implements an evaluator for stacked (independent) note estimations.
    """

    def __init__(self, offset_ratio=None, key=None, save_dir=None, patterns=None, verbose=False):
        """
        Initialize parameters for the evaluator.

        Parameters
        ----------
        See Evaluator class...

        offset_ratio : float
          Ratio of the reference note's duration used to define the offset tolerance
        """

        super().__init__(key, save_dir, patterns, verbose)

        self.offset_ratio = offset_ratio

    @staticmethod
    def get_default_key():
        """
        Default key for notes.
        """

        return tools.KEY_NOTES

    def unpack(self, data):
        """
        Unpack notes using the default notes key rather than the specified key.

        Parameters
        ----------
        data : object
          Presumably either a dictionary containing ground-truth
          or model output, or the already-unpacked notes

        Returns
        ----------
        data : object
          Unpacked notes or same object provided if no dictionary
        """

        # Determine the relevant key for evaluation
        key = self.get_default_key()

        # Check if a dictionary was provided and if the key is in the dictionary
        data = tools.try_unpack_dict(data, key)

        return data

    def evaluate(self, estimated, reference):
        """
        Evaluate stacked note estimates with respect to a reference.

        Parameters
        ----------
        estimated : dict
          Dictionary containing (slice -> (pitches, intervals)) pairs
        reference : dict
          Dictionary containing (slice -> (pitches, intervals)) pairs

        Returns
        ----------
        results : dict
          Dictionary containing precision, recall, and f-measure
        """

        # Initialize empty arrays to hold results for each degree of freedom
        precision, recall, f_measure = np.empty(0), np.empty(0), np.empty(0)

        # Loop through the stack of notes
        for key in estimated.keys():
            # Extract the loose note groups from the stack
            pitches_ref, intervals_ref = estimated[key]
            pitches_est, intervals_est = reference[key]

            # Convert notes to Hertz
            pitches_ref = tools.notes_to_hz(pitches_ref)
            pitches_est = tools.notes_to_hz(pitches_est)

            # Calculate frame-wise precision, recall, and f1 score with or without offset
            p, r, f, _ = evaluate_notes(ref_intervals=intervals_ref,
                                        ref_pitches=pitches_ref,
                                        est_intervals=intervals_est,
                                        est_pitches=pitches_est,
                                        offset_ratio=self.offset_ratio)

            # Add the results to the respective array
            precision = np.append(precision, p)
            recall = np.append(recall, r)
            f_measure = np.append(f_measure, f)

        # Package the results into a dictionary
        results = {
            tools.KEY_PRECISION : precision,
            tools.KEY_RECALL : recall,
            tools.KEY_F1 : f_measure
        }

        return results


class NoteEvaluator(StackedNoteEvaluator):
    """
    Implements an evaluator for notes.
    """

    def __init__(self, offset_ratio=None, key=None, save_dir=None, patterns=None, verbose=False):
        """
        Initialize parameters for the evaluator.

        Parameters
        ----------
        See StackedNoteEvaluator class...
        """

        super().__init__(offset_ratio, key, save_dir, patterns, verbose)

    def evaluate(self, estimated, reference):
        """
        Evaluate note estimates with respect to a reference.

        Parameters
        ----------
        estimated : ndarray (N x 3)
          Array of estimated note intervals and pitches by row
          N - number of notes
        reference : ndarray (N x 3)
          Array of ground-truth note intervals and pitches by row
          N - number of notes

        Returns
        ----------
        results : dict
          Dictionary containing precision, recall, and f-measure
        """

        # Convert the batches notes to notes
        notes_est = tools.batched_notes_to_notes(estimated)
        notes_ref = tools.batched_notes_to_notes(reference)

        # Convert the notes to stacked notes
        stacked_notes_est = tools.notes_to_stacked_notes(*notes_est)
        stacked_notes_ref = tools.notes_to_stacked_notes(*notes_ref)

        # Call the parent class evaluate function
        results = super().evaluate(stacked_notes_est, stacked_notes_ref)

        # Average the results across the degree of freedom - i.e. collapse extraneous dimension
        results = average_results(results)

        return results


class StackedPitchListEvaluator(Evaluator):
    """
    Implements an evaluator for stacked (independent) pitch list estimations.

    This is equivalent to the discrete multi pitch evaluation protocol for
    discrete estimates, but is more general and works for continuous pitch estimations.
    """

    def __init__(self, key=None, save_dir=None, patterns=None, verbose=False):
        """
        Initialize parameters for the evaluator.

        Parameters
        ----------
        See Evaluator class...
        """

        super().__init__(key, save_dir, patterns, verbose)

    @staticmethod
    def get_default_key():
        """
        Default key for pitch lists.
        """

        return tools.KEY_PITCHLIST

    def evaluate(self, estimated, reference):
        """
        Evaluate stacked pitch list estimates with respect to a reference.

        Parameters
        ----------
        estimated : dict
          Dictionary containing (slice -> (times, pitch_list)) pairs
        reference : dict
          Dictionary containing (slice -> (times, pitch_list)) pairs

        Returns
        ----------
        results : dict
          Dictionary containing precision, recall, and f-measure
        """

        # Initialize empty arrays to hold results for each degree of freedom
        precision, recall, f_measure = np.empty(0), np.empty(0), np.empty(0)

        # Loop through the stack of pitch lists
        for key in estimated.keys():
            # Extract the pitch lists from the stack
            times_ref, pitches_ref = estimated[key]
            times_est, pitches_est = reference[key]

            # Convert pitch lists to Hertz
            pitches_ref = tools.pitch_list_to_hz(pitches_ref)
            pitches_est = tools.pitch_list_to_hz(pitches_est)

            # Calculate frame-wise precision, recall, and f1 score for continuous pitches
            frame_metrics = evaluate_frames(times_ref, pitches_ref, times_est, pitches_est)

            # Extract observation-wise precision and recall
            p, r = frame_metrics['Precision'], frame_metrics['Recall']

            # Calculate the f1-score using the harmonic mean formula
            f = hmean([p + EPSILON, r + EPSILON]) - EPSILON

            # Add the results to the respective array
            precision = np.append(precision, p)
            recall = np.append(recall, r)
            f_measure = np.append(f_measure, f)

        # Package the results into a dictionary
        results = {
            tools.KEY_PRECISION : precision,
            tools.KEY_RECALL : recall,
            tools.KEY_F1 : f_measure
        }

        return results


class PitchListEvaluator(StackedPitchListEvaluator):
    """
    Evaluates pitch list estimates against a reference.

    This is equivalent to the discrete multi pitch evaluation protocol for
    discrete estimates, but is more general and works for continuous pitch estimations.
    """

    def __init__(self, key=None, save_dir=None, patterns=None, verbose=False):
        """
        Initialize parameters for the evaluator.

        Parameters
        ----------
        See Evaluator class...
        """

        super().__init__(key, save_dir, patterns, verbose)

    def evaluate(self, estimated, reference):
        """
        Evaluate pitch list estimates with respect to a reference.

        Parameters
        ----------
        estimated : tuple containing
          times : ndarray (N)
            Time in seconds of beginning of each frame
            N - number of time samples (frames)
          pitch_list : list of ndarray (N x [...])
            Array of pitches corresponding to notes
            N - number of pitch observations (frames)
        reference : tuple containing
          times : ndarray (N)
            Time in seconds of beginning of each frame
            N - number of time samples (frames)
          pitch_list : list of ndarray (N x [...])
            Array of pitches corresponding to notes
            N - number of pitch observations (frames)

        Returns
        ----------
        results : dict
          Dictionary containing precision, recall, and f-measure
        """

        # Convert the pitch lists to stacked pitch lists
        stacked_pitch_list_est = tools.pitch_list_to_stacked_pitch_list(*estimated)
        stacked_pitch_list_ref = tools.pitch_list_to_stacked_pitch_list(*reference)

        # Call the parent class evaluate function
        results = super().evaluate(stacked_pitch_list_est, stacked_pitch_list_ref)

        # Average the results across the degree of freedom - i.e. collapse extraneous dimension
        results = average_results(results)

        return results


class TablatureEvaluator(Evaluator):
    """
    Implements an evaluator for tablature.
    """

    def __init__(self, profile, key=None, save_dir=None, patterns=None, verbose=False):
        """
        Initialize parameters for the evaluator.

        Parameters
        ----------
        See Evaluator class for others...

        profile : InstrumentProfile (instrument.py)
          Instrument profile detailing experimental setup
        """

        super().__init__(key, save_dir, patterns, verbose)

        self.profile = profile

    @staticmethod
    def get_default_key():
        """
        Default key for tablature.
        """

        return tools.KEY_TABLATURE

    def pre_proc(self, estimated, reference):
        """
        By default, we anticipate neither estimate
        or reference to be in stacked multi pitch format.

        TODO - do something similar for pitch list wrapper reference

        Parameters
        ----------
        estimated : object
          Dictionary containing ground-truth or the already-unpacked entry
        reference : object
          Dictionary containing model output or the already-unpacked entry

        Returns
        ----------
        estimated : object
          Estimate relevant to the evaluation
        reference : object
          Reference relevant to the evaluation
        """

        # Unpacked estimate and reference if dictionaries were provided
        tablature_est, tablature_ref = super().pre_proc(estimated, reference)

        # Convert from tablature format to stacked multi pitch format
        tablature_est = tools.tablature_to_stacked_multi_pitch(tablature_est, self.profile)
        tablature_ref = tools.tablature_to_stacked_multi_pitch(tablature_ref, self.profile)

        return tablature_est, tablature_ref

    def evaluate(self, estimated, reference):
        """
        Evaluate a stacked multi pitch tablature estimate with respect to a reference.

        Parameters
        ----------
        estimated : ndarray (S x F x T)
          Array of multiple discrete pitch activation maps
          S - number of slices in stack
          F - number of discrete pitches
          T - number of frames
        reference : ndarray (S x F x T)
          Array of multiple discrete pitch activation maps
          Dimensions same as estimated

        Returns
        ----------
        results : dict
          Dictionary containing precision, recall, f-measure, and tdr
        """

        # Flatten the estimated and reference data along the pitch and degree-of-freedom axis
        flattened_tablature_est = estimated.flatten()
        flattened_tablature_ref = reference.flatten()

        # Count the number of activations predicted
        num_predicted = np.sum(flattened_tablature_est, axis=-1)
        # Count the number of activations referenced
        num_ground_truth = np.sum(flattened_tablature_ref, axis=-1)

        # Determine the number of correct tablature predictions,
        # where estimated activation lines up with reference
        num_correct_tablature = np.sum(flattened_tablature_est * flattened_tablature_ref, axis=-1)

        # Calculate precision and recall
        precision = num_correct_tablature / (num_predicted + EPSILON)
        recall = num_correct_tablature / (num_ground_truth + EPSILON)

        # Calculate the f1-score using the harmonic mean formula
        f_measure = hmean([precision + EPSILON, recall + EPSILON]) - EPSILON

        # Collapse the stacked multi pitch activations into a single representation
        multi_pitch_est = tools.stacked_multi_pitch_to_multi_pitch(estimated)
        multi_pitch_ref = tools.stacked_multi_pitch_to_multi_pitch(reference)

        # Flatten the estimated and reference multi pitch activations
        flattened_multi_pitch_est = multi_pitch_est.flatten()
        flattened_multi_pitch_ref = multi_pitch_ref.flatten()

        # Determine the number of correct predictions,
        # where estimated activation lines up with reference
        num_correct_multi_pitch = np.sum(flattened_multi_pitch_est * flattened_multi_pitch_ref, axis=-1)

        # Calculate the tablature disambiguation rate
        tdr = num_correct_tablature / (num_correct_multi_pitch + EPSILON)

        # Package the results into a dictionary
        results = {
            tools.KEY_PRECISION : precision,
            tools.KEY_RECALL : recall,
            tools.KEY_F1 : f_measure,
            tools.KEY_TDR : tdr
        }

        return results


class SoftmaxAccuracy(Evaluator):
    """
    Implements an evaluator for calculating accuracy of softmax groups.
    """

    def __init__(self, key, save_dir=None, patterns=None, verbose=False):
        """
        Initialize parameters for the evaluator.

        Parameters
        ----------
        See Evaluator class for others...
        """

        super().__init__(key, save_dir, patterns, verbose)

    @staticmethod
    def get_default_key():
        """
        A key must be provided for softmax groups accuracy.
        """

        return NotImplementedError

    def evaluate(self, estimated, reference):
        """
        Evaluate class membership estimates with respect to a reference.

        Parameters
        ----------
        estimated : ndarray (S x T)
          Array of class membership estimates for multiple degrees of freedom (e.g. strings)
          S - number of degrees of freedom
          T - number of samples or frames
        reference : ndarray (S x F x T)
          Array of class membership ground-truth
          Dimensions same as estimated

        Returns
        ----------
        results : dict
          Dictionary containing accuracy
        """

        # Determine the number of correctly identified classes across all groups
        num_correct = np.sum(estimated == reference)

        # Determine the total number of samples being evaluated
        num_total = reference.size

        # Calculate accuracy averaged over all softmax groups
        accuracy = num_correct / num_total

        # Package the results into a dictionary
        results = {
            tools.KEY_ACCURACY : accuracy
        }

        return results
