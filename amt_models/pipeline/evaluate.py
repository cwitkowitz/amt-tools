# My imports
from amt_models.pipeline import predict_multi
from amt_models.tools import pianoroll_to_pitchlist, to_single, arr_to_note_groups, to_multi, write_results
from amt_models.tools.constants import *

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


def average_results(d):
    d_avg = deepcopy(d)
    for key in d_avg.keys():
        if isinstance(d_avg[key], dict):
            d_avg[key] = average_results(d_avg[key])
        else:
            # Take the average of all entries and convert to float (necessary for logger)
            d_avg[key] = float(np.mean(d_avg[key]))
    return d_avg


def append_results(d1, d2):
    d3 = deepcopy(d1)
    for key in d2.keys():
        if key not in d3.keys():
            d3[key] = d2[key]
        elif isinstance(d2[key], dict):
            d3[key] = append_results(d3[key], d2[key])
        else:
            d3[key] = np.append(d3[key], d2[key])
    return d3


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
