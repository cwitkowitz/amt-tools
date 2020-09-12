# My imports
from pipeline.transcribe import *

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

def framewise_multi(t_ref, p_ref, t_est, p_est, low):
    multi_num = p_est.shape[0]

    assert multi_num == p_ref.shape[0]

    metrics = get_metrics_format()

    for i in range(multi_num):
        f_ref = pianoroll_to_pitchlist(p_ref[i], low)
        f_est = pianoroll_to_pitchlist(p_est[i], low)
        metrics = add_metric_dicts(metrics, framewise(t_ref, f_ref, t_est, f_est))

    metrics = average_metrics(metrics)
    return metrics

def framewise(t_ref, f_ref, t_est, f_est):
    # Compare the ground-truth to the predictions to get the frame-wise metrics
    frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)

    # Calculate frame-wise precision, recall, and f1 score
    f_pr, f_re = frame_metrics['Precision'], frame_metrics['Recall']
    f_f1 = hmean([f_pr + eps, f_re + eps]) - eps

    metrics = bundle_metrics(f_pr, f_re, f_f1)

    return metrics

def notewise_multi(ref, est, offset_ratio=None):
    multi_num = len(est)

    assert multi_num == len(ref)

    metrics = get_metrics_format()

    for i in range(multi_num):
        p_ref, i_ref = tuple(ref[i])
        p_est, i_est = tuple(est[i])
        metrics = add_metric_dicts(metrics, notewise(i_ref, p_ref, i_est, p_est, offset_ratio))

    metrics = average_metrics(metrics)
    return metrics

def notewise(i_ref, p_ref, i_est, p_est, offset_ratio=None):
    # Calculate frame-wise precision, recall, and f1 score with or without offset
    n_pr, n_re, n_f1, _ = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=offset_ratio)

    metrics = bundle_metrics(n_pr, n_re, n_f1)

    return metrics

def average_metrics(d):
    d_avg = deepcopy(d)
    for key in d_avg.keys():
        if len(d_avg[key]) == 0:
            d_avg.pop(key)
        else:
            d_avg[key] = float(np.mean(d_avg[key]))
    return d_avg

def average_results(d):
    d_avg = deepcopy(d)
    for key in d_avg.keys():
        if isinstance(d_avg[key], dict):
            d_avg[key] = average_metrics(d_avg[key])
        else:
            if len(d_avg[key]) > 0:
                d_avg[key] = float(np.mean(d_avg[key]))
            else:
                d_avg.pop(key)
    return d_avg

def add_metric_dicts(d1, d2):
    d3 = deepcopy(d1)
    for key in d3.keys():
        if isinstance(d2[key], list):
            d3[key] += d2[key]
        else:
            d3[key] += [d2[key]]
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
    # TODO - how necessary is this function - could I just create an empty dict?
    metrics = {'precision' : [],
               'recall' : [],
               'f1-score' : []
               }

    return metrics

def get_results_format():
    # TODO - how necessary is this function - could I just create an empty dict?
    # Create a dictionary to hold the metrics of each track
    results = {'pitch' : get_metrics_format(),
               #'pitch-tab' : get_metrics_format(),
               'note-on' : get_metrics_format(),
               'note-off' : get_metrics_format(),
               #'note-tab' : get_metrics_format(),
               'loss' : []}

    return results

def evaluate(prediction, reference, profile, log_dir=None, verbose=False):
    results = get_results_format()

    track_id = prediction['track']

    assert track_id == reference['track']

    if 'loss' in prediction.keys():
        results['loss'] = prediction['loss']

    # Add the frame-wise metrics to the dictionary
    pitch_ref = reference['pitch']
    f_ref = pianoroll_to_pitchlist(to_single(pitch_ref, profile), profile.low)
    t_ref = reference['times'][:-1]

    pitch_est = prediction['pitch_single']
    f_est = pianoroll_to_pitchlist(pitch_est, profile.low)
    t_est = prediction['times'][:-1]
    results['pitch'] = framewise(t_ref, f_ref, t_est, f_est)

    # Add the note-wise metrics to the dictionary
    p_est, i_est = prediction['notes_single']
    p_ref, i_ref = arr_to_note_groups(reference['notes'])
    results['note-on'] = notewise(i_ref, p_ref, i_est, p_est, offset_ratio=None)
    results['note-off'] = notewise(i_ref, p_ref, i_est, p_est, offset_ratio=0.2)

    if 'pitch_multi' in prediction.keys() and 'notes_multi' in prediction.keys():
        pitch_multi_ref = to_multi(pitch_ref, profile)
        pitch_multi_est = prediction['pitch_multi']
        pitch_tab_results = framewise_multi(t_ref, pitch_multi_ref, t_est, pitch_multi_est, profile.low)
        # TODO - TDR is not correct
        pitch_tdr = pitch_tab_results['precision'] / results['pitch']['precision'][0]
        pitch_tab_results['tdr'] = [pitch_tdr]
        results['pitch-tab'] = pitch_tab_results

        onsets = None
        if 'onsets' in reference.keys():
            onsets = to_multi(reference['onsets'], profile)
        notes_multi_ref = predict_multi(pitch_multi_ref, t_ref, profile.low, onsets)
        notes_multi_est = prediction['notes_multi']
        notes_tab_results = notewise_multi(notes_multi_ref, notes_multi_est)
        # TODO - TDR is not correct
        note_tdr = notes_tab_results['precision'] / results['note-on']['precision'][0]
        notes_tab_results['tdr'] = [note_tdr]
        results['note-tab'] = notes_tab_results

    if log_dir is not None:
        # Construct a path for the track's transcription and separation results
        results_path = os.path.join(log_dir, f'{track_id}.txt')
        write_results(results, results_path, verbose)

    if verbose:
        # Add a newline to the console
        print()

    return results
