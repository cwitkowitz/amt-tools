# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .inference import run_online, run_offline
from . import tools

# Regular imports
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.multipitch import evaluate as evaluate_frames
from abc import abstractmethod
from scipy.stats import hmean
from mir_eval import util
from copy import deepcopy

import numpy as np
import warnings
import torch
import json
import sys
import os

__all__ = [
    'validate',
    'average_results',
    'append_results',
    'log_results',
    'write_results',
    'pattern_match',
    'Evaluator',
    'ComboEvaluator',
    'LossWrapper',
    'StackedMultipitchEvaluator',
    'MultipitchEvaluator',
    'StackedNoteEvaluator',
    'NoteEvaluator',
    'StackedPitchListEvaluator',
    'PitchListEvaluator',
    'TablatureEvaluator',
    'SoftmaxAccuracy'
]

EPSILON = sys.float_info.epsilon

# TODO - what to do about null estimates? - mark as None and ignore in averaging?
#        - see https://stats.stackexchange.com/questions/8025/what-are-correct-values-for-precision-and-recall-when-the-denominators-equal-0

##################################################
# EVALUATION LOOP                                #
##################################################


def validate(model, dataset, evaluator, estimator=None, online=False):
    """
    Implements the validation or evaluation loop for a model and dataset partition.
    Optionally save predictions and log results.

    Parameters
    ----------
    model : TranscriptionModel
      Model to validate or evaluate
    dataset : TranscriptionDataset
      Dataset (partition) to use for validation or evaluation
    evaluator : Evaluator
      Evaluation protocol to use
    estimator : Estimator
      Estimation protocol to use
    online : bool
      Whether to evaluate the model in a mock-real-time fashion

    Returns
    ----------
    average : dict
      Dictionary containing all relevant results averaged across all tracks
    """

    # Turn off gradient computation
    with torch.no_grad():
        # Loop through the validation track ids
        for track_id in dataset.tracks:
            # Obtain the track data
            track_data = dataset.get_track_data(track_id)

            # Make sure the model is in evaluation mode, called here
            # in case there are any evaluation steps, such as resetting
            # language model state, to run before each track
            model.eval()

            if online:
                # Perform the inference step in mock-real-time fashion
                predictions = run_online(track_data, model, estimator)
            else:
                # Perform the inference step offline
                predictions = run_offline(track_data, model, estimator)

            # Evaluate the predictions and track the results
            evaluator.process_track(predictions, track_data, track_id)

    # Obtain the average results from this validation loop
    average = evaluator.average_results()

    return average


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


def log_results(results, writer, step=0, patterns=None, tag='', prnt=False):
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
    prnt : bool
      Whether to additionally print results to console
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

                if prnt:
                    # Print the results for the entry to the console
                    print(json.dumps({'iter': step, f'{tag}/{key}': entry}))


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

    def __init__(self, unpack_key=None, results_key=None, save_dir=None, patterns=None, verbose=False):
        """
        Initialize parameters common to all evaluators and instantiate.

        Parameters
        ----------
        unpack_key : string or None (optional)
          Key to use when unpacking data for the evaluation
        results_key : string or None (optional)
          Key to use when organizing results
        save_dir : string or None (optional)
          Directory where results for each track will be written
        patterns : list of string or None (optional)
          Only write/log metrics containing these patterns (e.g. ['f1', 'pr']) (None for all metrics)
        verbose : bool
          Whether to print any written text to console as well
        """

        # Set up the dictionary keys to use for extracting data and organizing results
        self.unpack_key = self.get_default_key() if unpack_key is None else unpack_key
        self.results_key = self.get_default_key() if results_key is None else results_key

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

    @staticmethod
    @abstractmethod
    def get_default_key():
        """
        Provide the default key to use in the event no key was provided.
        """

        return NotImplementedError

    def unpack(self, estimated, reference):
        """
        Attempt to unpack data relevant to the evaluator from the provided estimates/ground-truth.

        Note: This method can be overridden in order to insert extra steps, such
              as for manipulating the format of the estimate and/or ground-truth.

        Parameters
        ----------
        estimated : dict
          Dictionary containing estimate relevant to evaluation
        reference : dict
          Dictionary containing ground-truth relevant to evaluation

        Returns
        ----------
        estimated : object
          Estimate relevant to evaluation
        reference : object
          Ground-truth relevant to evaluation
        """

        # Unpack estimate and reference from provided dictionaries
        estimated = tools.unpack_dict(estimated, self.unpack_key)
        reference = tools.unpack_dict(reference, self.unpack_key)

        if estimated is None:
            # Estimate relevant to this evaluator does not exist or was not unpacked properly
            warnings.warn(f'Entry for key \'{self.unpack_key}\' ' +
                          f'not found in estimates.', category=RuntimeWarning)

        if reference is None:
            # Ground-truth relevant to this evaluator does not exist or was not unpacked properly
            warnings.warn(f'Entry for key \'{self.unpack_key}\' ' +
                          f'not found in ground-truth.', category=RuntimeWarning)

        return estimated, reference

    @abstractmethod
    def evaluate(self, estimated, reference):
        """
        Evaluate an estimate with respect to a reference.

        Parameters
        ----------
        estimated : object
          Estimate relevant to evaluation
        reference : object
          Ground-truth relevant to evaluation
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

    def process_track(self, estimated, reference, track=None):
        """
        Calculate the results, write them, and track them within the evaluator.

        Parameters
        ----------
        estimated : dict
          Dictionary containing estimate relevant to evaluation
        reference : dict
          Dictionary containing ground-truth relevant to evaluation
        track : string
          Name of the track being processed

        Returns
        ----------
        results : dictionary
          Dictionary containing results of tracks arranged by metric
        """

        # Unpack relevant data and calculate the results
        results = self.evaluate(*self.unpack(estimated, reference))

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
          All evaluators to run
        """

        self.evaluators = evaluators

        super().__init__(None, None, save_dir, patterns, verbose)

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
            if tools.query_dict(average, evaluator.results_key):
                # Add new entries to the results
                average[evaluator.results_key].update(results)
            else:
                # Create a new entry for the results
                average[evaluator.results_key] = results

        return average

    @staticmethod
    @abstractmethod
    def get_default_key():
        """
        This should not be called directly on a ComboEvaluator.
        """

        return NotImplementedError

    @abstractmethod
    def unpack(self, estimated, reference):
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

    def process_track(self, estimated, reference, track=None):
        """
        Very similar to parent method, except file is written after results are
        calculated for each evaluator and packaged into a single dictionary.

        Parameters
        ----------
        estimated : dict
          Dictionary containing estimates relevant to evaluation
        reference : dict
          Dictionary containing all ground-truth relevant to evaluation
        track : string
          Name of the track being processed

        Returns
        ----------
        results : dictionary
          Dictionary containing results of tracks arranged by metric
        """

        # Copy the raw output dictionary and use it to hold estimates
        results = dict()

        # Loop through the evaluators
        for evaluator in self.evaluators:
            # Unpack relevant data and calculate the results
            new_results = evaluator.evaluate(*evaluator.unpack(estimated, reference))

            # Check if there is already an entry for the evaluator's key
            if tools.query_dict(results, evaluator.results_key):
                # Add new entries to the results
                results[evaluator.results_key].update(new_results)
            else:
                # Create a new entry for the results
                results[evaluator.results_key] = new_results

            # Add the results to the tracked dictionary
            evaluator.results = append_results(evaluator.results, new_results)

        # Write the results
        self.write(results, track)

        return results


class LossWrapper(Evaluator):
    """
    Simple wrapper for tracking, writing, and logging loss.
    """

    @staticmethod
    def get_default_key():
        """
        Default key for loss.
        """

        return tools.KEY_LOSS

    def unpack(self, estimated, reference=None):
        """
        Unpack the loss from the dictionary of estimates and ignore the ground-truth.

        Parameters
        ----------
        estimated : dict
          Dictionary containing loss information
        reference : irrelevant

        Returns
        ----------
        loss : dict
          Dictionary containing computed loss terms
        reference : None
          Null pointer in place of unpacked ground-truth data
        """

        # Unpack loss from provided dictionary
        loss = tools.unpack_dict(estimated, self.unpack_key)

        if loss is None:
            # Loss does not exist or was not unpacked properly
            warnings.warn(f'Entry for key \'{self.unpack_key}\' ' +
                          f'not found in estimates.', category=RuntimeWarning)

        return loss, None

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

        # Pass the loss directly through
        results = estimated

        return results


class StackedEvaluator(Evaluator):
    """
    Implements an evaluator for stacked representations.
    """

    def __init__(self, average_slices=False, unpack_key=None, results_key=None,
                 save_dir=None, patterns=None, verbose=False):
        """
        Initialize parameters for the evaluator.

        Parameters
        ----------
        See Evaluator class for others...

        average_slices : bool
          Whether to collapse the slice dimension of the results by averaging
        """

        super().__init__(unpack_key, results_key, save_dir, patterns, verbose)

        self.average_slices = average_slices

    @staticmethod
    def average_slice_results(_results):
        """
        Average results split by slices of a stacked representation.

        Parameters
        ----------
        _results : dict
          Results dictionary with an entry for each slice

        Returns
        ----------
        results : dict
          Collapsed results dictionary
        """

        # Initialize a new dictionary to hold averages
        results = dict()

        # Loop through all tracked slices
        for key in _results.keys():
            # Append the results of each slice to the new dictionary
            results = append_results(results, _results[key])

        # Average the results across slice
        results = average_results(results)

        return results


class StackedMultipitchEvaluator(StackedEvaluator):
    """
    Implements an evaluator for stacked multi pitch activation maps, i.e.
    independent multi pitch estimations across degrees of freedom or instruments.
    """

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
          Dictionary containing (slice -> (precision, recall, and f-measure)) pairs
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

        # Obtain integer keys for the slices
        slice_keys = list(range(len(f_measure)))

        # Create a dictionary of results for each slice
        results = [{
            tools.KEY_PRECISION : precision[slc],
            tools.KEY_RECALL : recall[slc],
            tools.KEY_F1 : f_measure[slc]
        } for slc in slice_keys]

        # Package the results into a dictionary
        results = dict(zip(slice_keys, results))

        if self.average_slices:
            # Average the results across the slices
            results = self.average_slice_results(results)

        return results


class MultipitchEvaluator(StackedMultipitchEvaluator):
    """
    Implements an evaluator for multi pitch activation maps.
    """

    def __init__(self, unpack_key=None, results_key=None,
                 save_dir=None, patterns=None, verbose=False):
        """
        Initialize parameters for the evaluator.

        Parameters
        ----------
        See StackedEvaluator class...
        """

        super().__init__(True, unpack_key, results_key, save_dir, patterns, verbose)

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

        return results


class StackedNoteEvaluator(StackedEvaluator):
    """
    Implements an evaluator for stacked (independent) note estimations.
    """

    def __init__(self, offset_ratio=None, average_slices=False, unpack_key=None,
                 results_key=None, save_dir=None, patterns=None, verbose=False):
        """
        Initialize parameters for the evaluator.

        Parameters
        ----------
        See StackedEvaluator class for others...

        offset_ratio : float
          Ratio of the reference note's duration used to define the offset tolerance
        """

        super().__init__(average_slices, unpack_key, results_key, save_dir, patterns, verbose)

        self.offset_ratio = offset_ratio

    @staticmethod
    def get_default_key():
        """
        Default key for notes.
        """

        return tools.KEY_NOTES

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
          Dictionary containing (slice -> (precision, recall, and f-measure)) pairs
        """

        # Initialize an empty dictionary to hold results for each degree of freedom
        results = dict()

        # Obtain the keys of both the estimated and reference data
        keys_est, keys_ref = list(estimated.keys()), list(reference.keys())

        # Loop through the stack of notes
        for k in range(len(keys_ref)):
            # Extract the loose note groups from the stack
            pitches_est, intervals_est = estimated[keys_est[k]]
            pitches_ref, intervals_ref = reference[keys_ref[k]]

            # Convert notes to Hertz
            pitches_ref = tools.notes_to_hz(pitches_ref)
            pitches_est = tools.notes_to_hz(pitches_est)

            # Calculate precision, recall, and f1 score of matched notes with or without offset
            p, r, f, _ = evaluate_notes(ref_intervals=intervals_ref,
                                        ref_pitches=pitches_ref,
                                        est_intervals=intervals_est,
                                        est_pitches=pitches_est,
                                        offset_ratio=self.offset_ratio)

            # Package the results into a dictionary
            results.update({keys_est[k] : {
                tools.KEY_PRECISION : p,
                tools.KEY_RECALL : r,
                tools.KEY_F1 : f
            }})

        if self.average_slices:
            # Average the results across the slices
            results = self.average_slice_results(results)

        return results


class NoteEvaluator(StackedNoteEvaluator):
    """
    Implements an evaluator for notes.
    """

    def __init__(self, offset_ratio=None, unpack_key=None, results_key=None,
                 save_dir=None, patterns=None, verbose=False):
        """
        Initialize parameters for the evaluator.

        Parameters
        ----------
        See StackedNoteEvaluator...
        """

        super().__init__(offset_ratio, True, unpack_key, results_key, save_dir, patterns, verbose)

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

        return results


class StackedPitchListEvaluator(StackedEvaluator):
    """
    Implements an evaluator for stacked (independent) pitch list estimations.

    This is equivalent to the discrete multi pitch evaluation protocol for
    discrete estimates, but is more general and works for continuous pitch estimations.
    """

    def __init__(self, pitch_tolerances=None, average_slices=False, unpack_key=None,
                 results_key=None, save_dir=None, patterns=None, verbose=False):
        """
        Initialize parameters for the evaluator.

        Parameters
        ----------
        See StackedEvaluator class for others...

        pitch_tolerances : list of float
          Semitone tolerances for considering a frequency estimate correct relative to a reference
        """

        super().__init__(average_slices, unpack_key, results_key, save_dir, patterns, verbose)

        if pitch_tolerances is None:
            pitch_tolerances = [1/2]

        self.pitch_tolerances = pitch_tolerances

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
          Dictionary containing (slice -> (precision, recall, and f-measure)) pairs
        """

        # Obtain the keys of both the estimated and reference data
        keys_est, keys_ref = list(estimated.keys()), list(reference.keys())

        # Initialize an empty dictionary to hold results for each degree of freedom
        results = dict()

        # Loop through the stack of pitch lists
        for k in range(len(keys_ref)):
            # Extract the pitch lists from the stack
            times_est, pitches_est = estimated[keys_est[k]]
            times_ref, pitches_ref = reference[keys_ref[k]]

            # Convert pitch lists to Hertz
            pitches_ref = tools.pitch_list_to_hz(pitches_ref)
            pitches_est = tools.pitch_list_to_hz(pitches_est)

            for tol in self.pitch_tolerances:
                # Calculate frame-wise precision, recall, and f1 score for continuous pitches
                frame_metrics = evaluate_frames(ref_time=times_ref,
                                                ref_freqs=pitches_ref,
                                                est_time=times_est,
                                                est_freqs=pitches_est,
                                                window=tol)

                # Extract observation-wise precision and recall
                p, r = frame_metrics['Precision'], frame_metrics['Recall']

                # Calculate the f1-score using the harmonic mean formula
                f = hmean([p + EPSILON, r + EPSILON]) - EPSILON

                # Package the results into a dictionary
                results.update({keys_est[k]: {
                    f'{tol}' : {
                        tools.KEY_PRECISION : p,
                        tools.KEY_RECALL : r,
                        tools.KEY_F1 : f}
                }})

        if self.average_slices:
            # Average the results across the slices
            results = self.average_slice_results(results)

        return results


class PitchListEvaluator(StackedPitchListEvaluator):
    """
    Evaluates pitch list estimates against a reference.

    This is equivalent to the discrete multi pitch evaluation protocol for
    discrete estimates, but is more general and works for continuous pitch estimations.
    """

    def __init__(self, pitch_tolerances=None, unpack_key=None, results_key=None,
                 save_dir=None, patterns=None, verbose=False):
        """
        Initialize parameters for the evaluator.

        Parameters
        ----------
        See StackedPitchListEvaluator class...
        """

        super().__init__(pitch_tolerances, True, unpack_key, results_key, save_dir, patterns, verbose)

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

        return results


class TablatureEvaluator(Evaluator):
    """
    Implements an evaluator for tablature.
    """

    def __init__(self, profile, unpack_key=None, results_key=None,
                 save_dir=None, patterns=None, verbose=False):
        """
        Initialize parameters for the evaluator.

        Parameters
        ----------
        See Evaluator class for others...

        profile : InstrumentProfile (instrument.py)
          Instrument profile detailing experimental setup
        """

        super().__init__(unpack_key, results_key, save_dir, patterns, verbose)

        self.profile = profile

    @staticmethod
    def get_default_key():
        """
        Default key for tablature.
        """

        return tools.KEY_TABLATURE

    def evaluate(self, estimated, reference):
        """
        Evaluate a stacked multi pitch tablature estimate with respect to a reference.

        Parameters
        ----------
        estimated : ndarray (S x T)
          Array of class membership for multiple degrees of freedom (e.g. strings)
          S - number of strings or degrees of freedom
          T - number of frames
        reference : ndarray (S x T)
          Array of class membership for multiple degrees of freedom (e.g. strings)
          Dimensions same as estimated

        Returns
        ----------
        results : dict
          Dictionary containing precision, recall, f-measure, and tdr
        """

        # Convert from tablature format to logistic activations format
        tablature_est = tools.tablature_to_logistic(estimated, self.profile, silence=False)
        tablature_ref = tools.tablature_to_logistic(reference, self.profile, silence=False)

        # Flatten the estimated and reference data along the pitch and degree-of-freedom axis
        flattened_tablature_est = tablature_est.flatten()
        flattened_tablature_ref = tablature_ref.flatten()

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
        f_measure = util.f_measure(precision, recall)

        # Collapse the stacked multi pitch activations into a single representation
        multi_pitch_est = tools.stacked_multi_pitch_to_multi_pitch(
            tools.tablature_to_stacked_multi_pitch(estimated, self.profile))
        multi_pitch_ref = tools.stacked_multi_pitch_to_multi_pitch(
            tools.tablature_to_stacked_multi_pitch(reference, self.profile))

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

    @staticmethod
    def get_default_key():
        """
        Default key for tablature.
        """

        return tools.KEY_TABLATURE

    def evaluate(self, estimated, reference):
        """
        Evaluate class membership estimates with respect to a reference.

        Parameters
        ----------
        estimated : ndarray (S x T)
          Array of class membership estimates for multiple degrees of freedom (e.g. strings)
          S - number of degrees of freedom
          T - number of samples or frames
        reference : ndarray (S x T)
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
