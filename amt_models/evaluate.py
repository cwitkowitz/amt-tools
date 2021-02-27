# My imports
import amt_models.tools as tools

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


def framewise_multi(t_ref, p_ref, t_est, p_est, low):
    multi_num = p_est.shape[0]

    assert multi_num == p_ref.shape[0]

    metrics = {}

    for i in range(multi_num):
        f_ref = pianoroll_to_pitchlist(p_ref[i], low)
        f_est = pianoroll_to_pitchlist(p_est[i], low)
        metrics = append_results(metrics, framewise(t_ref, f_ref, t_est, f_est))

    metrics = average_results(metrics)

    return metrics


def framewise(t_ref, f_ref, t_est, f_est):
    # Compare the ground-truth to the predictions to get the frame-wise metrics
    frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)

    # Calculate frame-wise precision, recall, and f1 score
    pr, re = frame_metrics['Precision'], frame_metrics['Recall']
    f1 = hmean([pr + eps, re + eps]) - eps

    metrics = {
        PR_KEY : pr,
        RC_KEY : re,
        F1_KEY : f1
    }

    return metrics


def notewise_multi(ref, est, offset_ratio=None):
    multi_num = len(est)

    assert multi_num == len(ref)

    metrics = {}

    for i in range(multi_num):
        p_ref, i_ref = tuple(ref[i])
        p_est, i_est = tuple(est[i])
        metrics = append_results(metrics, notewise(i_ref, p_ref, i_est, p_est, offset_ratio))

    metrics = average_results(metrics)

    return metrics


def notewise(i_ref, p_ref, i_est, p_est, offset_ratio=None):
    # Calculate frame-wise precision, recall, and f1 score with or without offset
    pr, re, f1, _ = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=offset_ratio)

    metrics = {
        PR_KEY: pr,
        RC_KEY: re,
        F1_KEY: f1
    }

    return metrics


def evaluate(prediction, reference, profile, log_dir=None, verbose=False):
    results = {}

    track_id = prediction[TR_ID]

    assert track_id == reference[TR_ID]

    if 'loss' in prediction.keys():
        results['loss'] = prediction['loss']

    # Frame-wise Multi-pitch Metrics
    pitch_ref = reference[PITCH]
    f_ref = pianoroll_to_pitchlist(to_single(pitch_ref, profile), profile.low)
    t_ref = reference[TIMES]

    pitch_est = prediction[SOLO_PITCH]
    f_est = pianoroll_to_pitchlist(pitch_est, profile.low)
    t_est = prediction[TIMES]

    # Add the frame-wise metrics to the dictionary
    results[PITCH] = framewise(t_ref[:-1], f_ref, t_est[:-1], f_est)

    # Note-wise Multi-pitch Metrics
    p_est, i_est = prediction[SOLO_NOTES]
    p_ref, i_ref = arr_to_note_groups(reference[NOTES])

    # Add the note-wise metrics to the dictionary
    results[NOTE_ON] = notewise(i_ref, p_ref, i_est, p_est, offset_ratio=None)
    results[NOTE_OFF] = notewise(i_ref, p_ref, i_est, p_est, offset_ratio=0.2)

    if MULT_PITCH in prediction.keys() and MULT_NOTES in prediction.keys():
        # Frame-wise Tab-pitch Metrics
        pitch_multi_ref = to_multi(pitch_ref, profile)
        pitch_multi_est = prediction[MULT_PITCH]
        pitch_tab_results = framewise_multi(t_ref[:-1], pitch_multi_ref, t_est[:-1], pitch_multi_est, profile.low)
        # TODO - TDR is not correct
        if results[PITCH][PR_KEY]:
            pitch_tdr = pitch_tab_results[PR_KEY] / results[PITCH][PR_KEY]
        else:
            pitch_tdr = 0
        pitch_tab_results[TDR] = pitch_tdr
        results[TAB_PITCH] = pitch_tab_results

        # Note-wise Tab-pitch Metrics
        onsets = None
        if ONSET in reference.keys():
            onsets = to_multi(reference[ONSET], profile)
        notes_multi_ref = predict_multi(pitch_multi_ref, t_ref, profile.low, onsets)
        notes_multi_est = prediction[MULT_NOTES]
        notes_tab_results = notewise_multi(notes_multi_ref, notes_multi_est)
        # TODO - TDR is not correct
        if results[NOTE_ON][PR_KEY] != 0:
            note_tdr = notes_tab_results[PR_KEY] / results[NOTE_ON][PR_KEY]
        else:
            note_tdr = 0
        notes_tab_results[TDR] = note_tdr
        results[TAB_NOTES] = notes_tab_results

    if log_dir is not None:
        # Construct a path for the track's transcription and separation results
        results_path = os.path.join(log_dir, f'{track_id}.txt')
        write_results(results, results_path, verbose)

    if verbose:
        # Add a newline to the console
        print()

    return results


class Evaluator(object):
    """
    Implements a generic music information retrieval evaluator.
    """

    def __init__(self, results_dir, patterns, verbose):
        """
        Initialize parameters common to all evaluators and instantiate.

        Parameters
        ----------
        results_dir : string or None (optional)
          Directory where results for each track will be written
        patterns : list of string or None (optional)
          Only write/log metrics containing these patterns (e.g. ['f1', 'pr']) (None for all metrics)
        verbose : bool
          Whether to print any written text to console as well
        """

        self.results_dir = results_dir
        self.patterns = patterns
        self.verbose = verbose

        if self.results_dir is not None:
            # Create the specified directory if it does not already exist
            os.makedirs(self.results_dir, exist_ok=True)

        # Initialize dictionary to track results
        self.results = None
        self.reset_results()

    def reset_results(self):
        """
        Reset tracked results to empty dictionary.
        """

        self.results = {}

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
    def get_key():
        """
        Provide the default key to use in extracting predictions and ground-truth.
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
        Evaluate one estimate with respect to a reference.

        Parameters
        ----------
        estimated : object
          Estimate relevant to the evaluation or the dictionary containing it
        reference : object
          Reference relevant to the evaluation or the dictionary containing it
        """

        return NotImplementedError

    def track_results(self, estimated, reference, track=None):
        # Make sure the estimated and reference data are unpacked
        estimated, reference = self.pre_proc(estimated, reference)

        # Calculate the results
        new_results = self.evaluate(estimated, reference)

        # Add the results to the tracked dictionary
        self.results = append_results(self.results, new_results)

        if self.results_dir is not None:
            # Determine how to name the results
            tag = tools.get_tag(track)

            # Construct a path for the results
            results_path = os.path.join(self.results_dir, f'{tag}.{tools.TXT_EXT}')

            # Open a file at the path with writing permissions
            with open(results_path, 'w') as results_file:
                # Write the results to a text file
                write_results(new_results, results_file, self.patterns, self.verbose)

    def finalize(self, writer, step=0):
        # Average the currently tracked results
        average = self.average_results()

        # Log the currently tracked results
        log_results(average, writer, step, patterns=self.patterns)

        # Reset the tracked results
        self.reset_results()


class StackedMultipitchEvaluator(Evaluator):
    """
    Implements an evaluator for stacked multi pitch activation maps, i.e.
    independent multi pitch estimations across degrees of freedom or instruments.
    """

    def __init__(self, results_dir=None, patterns=None, verbose=False):
        """
        Initialize parameters for the evaluator.

        Parameters
        ----------
        See Evaluator class...
        """

        super().__init__(results_dir, patterns, verbose)

    @staticmethod
    def get_key():
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

        Returns
        ----------
        results : dict
          Dictionary containing precision, recall, and f-measure
        """

        # Determine the shape necessary to flatten the last two dimensions
        flatten_shape = estimated.shape[:-2] + tuple([-1])

        # Flatten the estimated and reference data
        estimated = np.reshape(estimated, flatten_shape)
        reference = np.reshape(reference, flatten_shape)

        # Determine the number of correct predictions,
        # where estimated activation lines up with reference
        num_correct = np.sum(estimated * reference, axis=-1)

        # Count the number of activations predicted
        num_predicted = np.sum(estimated, axis=-1)
        # Count the number of activations referenced
        num_ground_truth = np.sum(reference, axis=-1)

        # Calculate precision and recall
        precision = num_correct / (num_predicted + EPSILON)
        recall = num_correct / (num_ground_truth + EPSILON)

        # Calculate the f1-score using the harmonic mean formula
        f_measure = hmean([precision + EPSILON, recall + EPSILON]) - EPSILON

        # TODO - what will break if I don't average across dof?
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

    def __init__(self, results_dir=None, patterns=None, verbose=False):
        """
        Initialize parameters for the evaluator.

        Parameters
        ----------
        See Evaluator class...
        """

        super().__init__(results_dir, patterns, verbose)

    def evaluate(self, estimated, reference):
        """
        Evaluate a stacked multi pitch estimate with respect to a reference.

        Parameters
        ----------
        estimated : ndarray (F x T)
          Predicted discrete pitch activation map
          F - number of discrete pitches
          T - number of frames
        reference : ndarray (F x T)
          Ground-truth discrete pitch activation map

        Returns
        ----------
        results : dict
          Dictionary containing precision, recall, and f-measure
        """

        # Convert the multi pitch arrays to stacked multi pitch arrays
        estimated = tools.multi_pitch_to_stacked_multi_pitch(estimated)
        reference = tools.multi_pitch_to_stacked_multi_pitch(reference)

        # Call the parent class evaluate function. Multi pitch is just a special
        # case of stacked multi pitch, where there is only one degree of freedom
        results = super().evaluate(estimated, reference)

        # Average the results across the degree of freedom - i.e. collapse extraneous dimension
        results = average_results(results)

        return results


class StackedNoteEvaluator(Evaluator):
    """
    Implements an evaluator for stacked (independent) note estimations.
    """

    def __init__(self, offset_ratio=None, results_dir=None, patterns=None, verbose=False):
        """
        Initialize parameters for the evaluator.

        Parameters
        ----------
        See Evaluator class...

        offset_ratio : float
          Ratio of the reference note's duration used to define the offset tolerance
        """

        super().__init__(results_dir, patterns, verbose)

        self.offset_ratio = offset_ratio

    @staticmethod
    def get_key():
        """
        Default key for notes.
        """

        return tools.KEY_NOTES

    def evaluate(self, estimated, reference):
        """
        Evaluate stacked note estimates with respect to a reference.

        Parameters
        ----------
        estimated :
        reference :

        Returns
        ----------
        results : dict
          Dictionary containing precision, recall, and f-measure
        """

        # Make sure the estimated and reference data are unpacked
        estimated, reference = self.pre_proc(estimated, reference)

        precision, recall, f_measure = np.empty(0), np.empty(0), np.empty(0)

        # Loop through the stack of notes
        for key in estimated.keys():
            pitches_ref, intervals_ref = reference[key]
            pitches_est, intervals_est = estimated[key]

            pitches_ref = tools.notes_to_hz(pitches_ref)
            pitches_est = tools.notes_to_hz(pitches_est)

            # Calculate frame-wise precision, recall, and f1 score with or without offset
            p, r, f, _ = evaluate_notes(ref_intervals=intervals_ref,
                                        ref_pitches=pitches_ref,
                                        est_intervals=intervals_est,
                                        est_pitches=pitches_est,
                                        offset_ratio=self.offset_ratio)

            precision = np.append(precision, p)
            recall = np.append(recall, r)
            f_measure = np.append(f_measure, f)

        # TODO - what will break if I don't average across dof?
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

    def __init__(self, offset_ratio=None, results_dir=None, patterns=None, verbose=False):
        """
        Initialize parameters for the evaluator.

        Parameters
        ----------
        See StackedNoteEvaluator class...
        """

        super().__init__(offset_ratio, results_dir, patterns, verbose)

    def evaluate(self, estimated, reference):
        """
        Evaluate note estimates with respect to a reference.

        Parameters
        ----------
        estimated :
        reference :

        Returns
        ----------
        results : dict
          Dictionary containing precision, recall, and f-measure
        """

        # Make sure the estimated and reference data are unpacked
        estimated, reference = self.pre_proc(estimated, reference)

        # TODO - should I be expecting batched notes or note groups?

        # Convert the batches notes to notes
        estimated = tools.batched_notes_to_notes(estimated)
        reference = tools.batched_notes_to_notes(reference)

        # Convert the notes to stacked notes
        estimated = tools.notes_to_stacked_notes(*estimated)
        reference = tools.notes_to_stacked_notes(*reference)

        # Call the parent class evaluate function
        results = super().evaluate(estimated, reference)

        # Average the results across the degree of freedom - i.e. collapse extraneous dimension
        results = average_results(results)

        return results


class PitchListEvaluator():
    pass


class ComboEvaluator():
    pass


##################################################
# RESULTS DICTIONARY                             #
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

    # TODO - average along axis 0 for stacked representations?

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


def log_results(results, writer, step=0, patterns=None):
    for type in results.keys():
        if isinstance(results[type], dict):
            for metric in results[type].keys():
                if pattern_match(metric, patterns):
                    writer.add_scalar(f'val/{type}/{metric}', results[type][metric], global_step=step)
        else:
            if patterns is None or type in patterns:
                writer.add_scalar(f'val/{type}', results[type], global_step=step)


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
            tools.write_and_print(file, f'-----{key}-----\n', verbose)
            # Call this function recursively
            write_results(results[key, file, verbose])
        else:
            # Check if the key matches the specified patterns
            if pattern_match(key, patterns):
                # Write the metric and corresponding result to the file
                tools.write_and_print(file, f' {key} : {results[key]}\n', verbose)

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
