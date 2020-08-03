# My imports
from tools.conversion import *
from tools.utils import *
from tools.io import *

# Regular imports
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.multipitch import evaluate as evaluate_frames
from scipy.stats import hmean
from copy import deepcopy

import numpy as np
import sys
import os

eps = sys.float_info.epsilon

# TODO - significant cleanup

def framewise(prediction, reference, hop_length, sample_rate):
    if 'tabs' in prediction:
        tabs_ref = np.transpose(reference['tabs'], (2, 0, 1))
        tabs_ref = np.argmax(tabs_ref, axis=-1).T

        f_ref = pianoroll_to_pitchlist(tabs_to_pianoroll(tabs_ref))
    else:
        f_ref = pianoroll_to_pitchlist(reference['frames'])
    t_ref = librosa.frames_to_time(range(len(f_ref)), sample_rate, hop_length)

    f_est = pianoroll_to_pitchlist(prediction['pianoroll'])
    # TODO - why a slight difference?
    t_est = librosa.frames_to_time(range(len(f_est)), sample_rate, hop_length)

    # Compare the ground-truth to the predictions to get the frame-wise metrics
    frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)

    # Calculate frame-wise precision, recall, and f1 score
    f_pr, f_re = frame_metrics['Precision'], frame_metrics['Recall']
    f_f1 = hmean([f_pr + eps, f_re + eps]) - eps

    metrics = bundle_metrics(f_pr, f_re, f_f1)

    return metrics

def notewise(prediction, reference, offset_ratio=None):
    p_est, i_est = arr_to_note_groups(prediction['notes'])
    p_ref, i_ref = arr_to_note_groups(reference['notes'])

    # Calculate frame-wise precision, recall, and f1 score with or without offset
    n_pr, n_re, n_f1, _ = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=offset_ratio)

    metrics = bundle_metrics(n_pr, n_re, n_f1)

    return metrics

def average_metrics(d):
    d_avg = deepcopy(d)
    for key in d_avg.keys():
        d_avg[key] = float(np.mean(d_avg[key]))
    return d_avg

def average_results(d):
    d_avg = deepcopy(d)
    for type in d_avg.keys():
        if isinstance(d_avg[type], dict):
            d_avg[type] = average_metrics(d_avg[type])
        else:
            d_avg[type] = float(np.mean(d_avg[type]))
    return d_avg

def add_metric_dicts(d1, d2):
    d3 = deepcopy(d1)
    for key in d3.keys():
        d3[key] += d2[key]
    return d3

def add_result_dicts(d1, d2):
    d3 = deepcopy(d1)
    for type in d3.keys():
        if isinstance(d3[type], dict):
            d3[type] = add_metric_dicts(d3[type], d2[type])
        else:
            d3[type] += [d2[type]]
    return d3

def bundle_metrics(pr, re, f1):
    metrics = get_metrics_format()
    metrics['precision'] += [pr]
    metrics['recall'] += [re]
    metrics['f1-score'] += [f1]
    return metrics

def get_metrics_format():
    metrics = {'precision' : [],
               'recall' : [],
               'f1-score' : []}

    return metrics

def get_results_format():
    # Create a dictionary to hold the metrics of each track
    results = {'frame' : get_metrics_format(),
               'note-on' : get_metrics_format(),
               'note-off' : get_metrics_format(),
               'loss' : []}

    return results

def evaluate(prediction, reference, hop_length, sample_rate, log_dir=None, verbose=False):
    results = get_results_format()

    track_id = prediction['track']

    assert track_id == reference['track']

    results['loss'] = prediction['loss']

    # Add the frame-wise metrics to the dictionary
    results['frame'] = framewise(prediction, reference, hop_length, sample_rate)

    # Add the note-wise metrics to the dictionary
    results['note-on'] = notewise(prediction, reference, offset_ratio=None)
    results['note-off'] = notewise(prediction, reference, offset_ratio=0.2)

    # TODO - tablature metrics

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        # Construct a path for the track's transcription and separation results
        results_path = os.path.join(log_dir, f'{track_id}.txt')
        # Open the file with writing permissions
        results_file = open(results_path, 'w')
        for type in results.keys():
            write_and_print(results_file, f'-----{type}-----\n', verbose)
            if isinstance(results[type], dict):
                for metric in results[type].keys():
                    write_and_print(results_file, f' {metric} : {results[type][metric]}\n', verbose)
            else:
                write_and_print(results_file, f' {type} : {results[type]}\n', verbose)
            write_and_print(results_file, '', verbose, '\n')
        # Close the results file
        results_file.close()

    if verbose:
        # Add a newline to the console
        print()

    return results
