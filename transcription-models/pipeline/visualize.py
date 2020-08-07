# My imports
from constants import *
from utils import *

from sacred import Experiment

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import librosa
import mirdata
import math
import jams
import os

plt.rcParams.update({'font.size': 16})

ex = Experiment('Visualize_Results')

# TODO - this requires a significant overhaul - it is really old


# TODO - put style values in constants.py or in config - yeah actually in config function which passes args = {}
# TODO - offsets are dependent on scale
# TODO - padding before/after is more/less significant depending on the difference between bounds
# TODO - colormap longer in y direction?

@ex.config
def config():
    # Visualize results for this track
    track_id = '00_BN1-129-Eb_comp'

    # Time boundaries for results in seconds
    # Example - [0, math.inf] for entire audio clip
    t_bounds = [0, 50]

    hop_len = 512

def pitch_contour(track_id, t_est, f_est, t_ref, f_ref, t_bounds, save_path=None):
    t_ref, f_ref = pitches_to_arr(t_ref, f_ref)
    t_est, f_est = pitches_to_arr(t_est, f_est)

    if t_est.size == 0:
        t_bounds = [t_bounds[0], min(t_bounds[1], t_ref[-1])]
    else:
        t_bounds = [t_bounds[0], min(t_bounds[1], max(t_est[-1], t_ref[-1]))]

    plt.figure()

    plt.scatter(t_ref, librosa.hz_to_midi(f_ref), color='black', label='Ref.', s=5)
    plt.scatter(t_est, librosa.hz_to_midi(f_est), color='orange', label='Est.', s=5, alpha=0.75)

    plt.title(track_id)
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch Contour (MIDI)')
    plt.legend(loc='upper right', markerscale=3, framealpha=0.5)
    plt.xlim(t_bounds[0] - 0.25, t_bounds[1] + 0.25)

    plt.gcf().set_size_inches(16, 4.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=500)
    else:
        plt.show()

def pianoroll(track_id, i_est, p_est, i_ref, p_ref, t_bounds, save_path=None):
    est_max, ref_max = 0, 0
    for i in range(NUM_STRINGS):
        if i_est.size != 0:
            est_max = max(est_max, np.max(i_est))
        if i_ref.size != 0:
            ref_max = max(ref_max, np.max(i_ref))
    t_bounds = [t_bounds[0], min(t_bounds[1], max(est_max, ref_max))]

    plt.figure()

    for s in range(NUM_STRINGS):
        for n in range(p_ref.size):
            t_st = i_ref[n][0]
            t_fn = i_ref[n][1]
            m_fq = int(round(librosa.hz_to_midi(p_ref[n])))

            plt.plot([t_st, t_fn], [m_fq] * 2, linewidth=10, color='black', label='Ref.')

        for n in range(p_est.size):
            t_st = i_est[n][0]
            t_fn = i_est[n][1]
            m_fq = int(round(librosa.hz_to_midi(p_est[n])))

            plt.plot([t_st, t_fn], [m_fq] * 2, linewidth=10, color='orange', label='Est.',alpha=0.75)

    handles = [mlines.Line2D([], [], color='black', linestyle='-', label='Ref.', linewidth=10),
               mlines.Line2D([], [], color='orange', linestyle='-', label='Est.', linewidth=10)]

    # The lowest possible note - i.e. the open note of the lowest string
    m_lw = librosa.note_to_midi(TUNING[0])

    # The highest possible note - i.e. the maximum fret on the highest string
    m_hg = librosa.note_to_midi(TUNING[NUM_STRINGS - 1]) + NUM_FRETS

    plt.title(track_id)
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch (MIDI)')
    plt.legend(handles=handles, loc='upper right', framealpha=0.5)
    plt.xlim(t_bounds[0] - 0.25, t_bounds[1] + 0.25)
    plt.ylim(m_lw - 1, m_hg + 1)

    plt.gcf().set_size_inches(16, 9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=500)
    else:
        plt.show()


def guitar_tabs(track_id, tabs_est, tabs_ref, t_bounds, offset=True, save_path=None):
    est_max, ref_max = 0, 0
    for i in range(NUM_STRINGS):
        if tabs_est[i][1].size != 0:
            est_max = max(est_max, np.max(tabs_est[i][1]))
        if tabs_ref[i][1].size != 0:
            ref_max = max(ref_max, np.max(tabs_ref[i][1]))
    t_bounds = [t_bounds[0], min(t_bounds[1], max(est_max, ref_max))]

    plt.figure()

    for s in range(NUM_STRINGS):
        p_ref = tabs_ref[s][0]
        i_ref = tabs_ref[s][1]
        for n in range(p_ref.size):
            t_st = i_ref[n][0]
            t_fn = i_ref[n][1]
            m_fq = librosa.hz_to_midi(p_ref[n])
            fret = int(round(m_fq - librosa.note_to_midi(TUNING[s])))

            plt.scatter(t_st, s + 1, marker="${}$".format(fret), color='black', label='Ref.', s=200)

            if offset:
                plt.plot([t_st + 0.03, t_fn - 0.03], [s + 1] * 2, linestyle='-', color='black', label='Ref.')

        p_est = tabs_est[s][0]
        i_est = tabs_est[s][1]
        for n in range(p_est.size):
            t_st = i_est[n][0]
            t_fn = i_est[n][1]
            m_fq = librosa.hz_to_midi(p_est[n])
            fret = int(round(m_fq - librosa.note_to_midi(TUNING[s])))

            plt.scatter(t_st, s + 1, marker="${}$".format(fret), color='orange', label='Est.', s=200, alpha=0.75)

            if offset:
                plt.plot([t_st + 0.03, t_fn - 0.03], [s + 1] * 2, linestyle='-', color='orange', label='Est.', alpha=0.75)

    handles = [mlines.Line2D([], [], color='black', linestyle='-', label='Ref.', linewidth=3),
               mlines.Line2D([], [], color='orange', linestyle='-', label='Est.', linewidth=3)]

    plt.title(track_id)
    plt.xlabel('Time (s)')
    plt.ylabel('String')
    plt.yticks(range(1, 7), TUNING)
    plt.legend(handles=handles, loc='upper right', framealpha=0.5)
    plt.xlim(t_bounds[0] - 0.25, t_bounds[1] + 0.25)

    plt.gcf().set_size_inches(16, 4.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=500)
    else:
        plt.show()

@ex.automain
def main(track_id, t_bounds, hop_len):
    # Create the dictionary directory if it does not already exist
    reset_generated_dir(GEN_FIGS_DIR, [f'{track_id}'], True)

    figs_dir = os.path.join(GEN_FIGS_DIR, f'{track_id}')

    t_est, f_est, t_ref, f_ref = get_frames_contours(track_id, hop_len)

    pitch_path = os.path.join(figs_dir, f'contours.jpg')
    pitch_contour(track_id, t_est, f_est, t_ref, f_ref, t_bounds, pitch_path)

    tabs_est, tabs_ref = get_tabs(track_id)

    tabs_path = os.path.join(figs_dir, f'tablature.jpg')
    guitar_tabs(track_id, tabs_est, tabs_ref, t_bounds, True, tabs_path)

    i_est, p_est, i_ref, p_ref = get_notes(track_id)

    proll_path = os.path.join(figs_dir, f'pianoroll.jpg')
    pianoroll(track_id, i_est, p_est, i_ref, p_ref, t_bounds, proll_path)
