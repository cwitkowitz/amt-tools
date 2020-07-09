"""
Get average scores and various other metrics across entire dataset
"""

# My imports
from tools.constants import *
from tools.datasets import *
from tools.dataproc import *
from tools.utils import *

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

def framewise(track, estim_dir, hop_length):
    track_id = track['track'][0]

    # Get the path to the frame-wise pitch estimations
    frm_txt_path = os.path.join(estim_dir, 'frames', f'{track_id}.txt')
    # Load the frame-wise pitch estimations
    t_est, f_est = load_ragged_time_series(frm_txt_path)

    tabs = track['tabs'][0].numpy()
    tabs = np.transpose(tabs, (2, 0, 1))
    tabs = np.argmax(tabs, axis=-1).T

    f_ref = pianoroll_to_pitchlist(tabs_to_pianoroll(tabs))
    # TODO - why a slight difference?
    t_ref = librosa.frames_to_time(range(len(f_ref)), SAMPLE_RATE, hop_length)

    # Compare the ground-truth to the predictions to get the frame-wise metrics
    frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)

    # Calculate frame-wise precision, recall, and f1 score
    f_pr, f_re = frame_metrics['Precision'], frame_metrics['Recall']
    f_f1 = hmean([f_pr + eps, f_re + eps]) - eps

    metrics = bundle_metrics(f_pr, f_re, f_f1)

    return metrics

def notewise(track, estim_dir, offset_ratio=None):
    track_id = track['track'][0]

    # Get the path to the note-wise estimations
    nte_txt_path = os.path.join(estim_dir, 'notes', f'{track_id}.txt')
    # Load the note-wise estimations
    i_est, p_est = load_valued_intervals(nte_txt_path)

    notes = track['notes'][0].numpy()
    i_ref, p_ref = notes[:, :2], notes[:, -1]

    # Calculate frame-wise precision, recall, and f1 score with or without offset
    n_pr, n_re, n_f1, _ = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=offset_ratio)

    metrics = bundle_metrics(n_pr, n_re, n_f1)

    return metrics

def add_metric_dicts(d1, d2):
    d3 = deepcopy(d1)
    for key in d3.keys():
        d3[key] += d2[key]
    return d3

def bundle_metrics(pr, re, f1):
    metrics = get_metrics_format()
    metrics['Precision'] += [pr]
    metrics['Recall'] += [re]
    metrics['F1-Score'] += [f1]
    return metrics

def get_metrics_format():
    metrics = {'Precision' : [],
               'Recall' : [],
               'F1-Score' : []}
    return metrics

def evaluate(loader, hop_length, estim_dir, log_dir, verbose):
    # Create a dictionary to hold the metrics of each track
    results = {'Frame' : get_metrics_format(),
               'Note-on' : get_metrics_format(),
               'Note-off' : get_metrics_format()}

    os.makedirs(log_dir, exist_ok=True)

    for idx, track in enumerate(loader):
        track_id = track['track'][0]

        # Construct a path for the track's transcription and separation results
        results_path = os.path.join(log_dir, f'{track_id}.txt')
        # Open the file with writing permissions
        results_file = open(results_path, 'w')

        # Add heading to file
        write_and_print(results_file, f'Evaluating track : {track_id}\n', verbose)

        # Add the frame-wise metrics to the dictionary
        frame_metrics = framewise(track, estim_dir, hop_length)
        results['Frame'] = add_metric_dicts(results['Frame'], frame_metrics)

        # Add the note-wise metrics to the dictionary
        note_metrics1 = notewise(track, estim_dir, offset_ratio=None)
        results['Note-on'] = add_metric_dicts(results['Note-on'], note_metrics1)
        note_metrics2 = notewise(track, estim_dir, offset_ratio=0.2)
        results['Note-off'] = add_metric_dicts(results['Note-off'], note_metrics2)

        # TODO - tablature metrics

        for type in results.keys():
            write_and_print(results_file, f'-----{type}-----\n', verbose)
            for metric in results[type].keys():
                write_and_print(results_file, f' {metric} : {results[type][metric][idx]}\n', verbose)
            write_and_print(results_file, '', verbose, '\n')

        # Close the results file
        results_file.close()

        if verbose:
            # Add a newline to the console
            print()

    # Obtain the average value across tracks for each metric
    for type in results.keys():
        for metric in results[type].keys():
            results[type][metric] = float(np.mean(results[type][metric]))

    return results
