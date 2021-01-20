# My imports
from amt_models.tools.instrument import *

# Regular imports
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import librosa

# TODO - see earlier commits to get started

profile = GuitarProfile()


def pianoroll(track_name, i_est, p_est, i_ref, p_ref, t_bounds, save_path=None):
    est_max, ref_max = 0, 0
    for i in range(profile.num_strings):
        if i_est.size != 0:
            est_max = max(est_max, np.max(i_est))
        if i_ref.size != 0:
            ref_max = max(ref_max, np.max(i_ref))
    t_bounds = [t_bounds[0], min(t_bounds[1], max(est_max, ref_max))]

    plt.figure()

    for s in range(profile.num_strings):
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
    m_lw = librosa.note_to_midi(profile.tuning[0])

    # The highest possible note - i.e. the maximum fret on the highest string
    m_hg = librosa.note_to_midi(profile.tuning[profile.num_strings - 1]) + profile.num_frets

    plt.title(track_name)
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


def guitar_tabs(track_name, tabs_est, tabs_ref, t_bounds, offset=True, save_path=None):
    est_max, ref_max = 0, 0
    for i in range(profile.num_strings):
        if tabs_est[i][1].size != 0:
            est_max = max(est_max, np.max(tabs_est[i][1]))
        if tabs_ref[i][1].size != 0:
            ref_max = max(ref_max, np.max(tabs_ref[i][1]))
    t_bounds = [t_bounds[0], min(t_bounds[1], max(est_max, ref_max))]

    plt.figure()

    for s in range(profile.num_strings):
        p_ref = tabs_ref[s][0]
        i_ref = tabs_ref[s][1]
        for n in range(p_ref.size):
            t_st = i_ref[n][0]
            t_fn = i_ref[n][1]
            m_fq = librosa.hz_to_midi(p_ref[n])
            fret = int(round(m_fq - librosa.note_to_midi(profile.tuning[s])))

            plt.scatter(t_st, s + 1, marker="${}$".format(fret), color='black', label='Ref.', s=200)

            if offset:
                plt.plot([t_st + 0.03, t_fn - 0.03], [s + 1] * 2, linestyle='-', color='black', label='Ref.')

        p_est = tabs_est[s][0]
        i_est = tabs_est[s][1]
        for n in range(p_est.size):
            t_st = i_est[n][0]
            t_fn = i_est[n][1]
            m_fq = librosa.hz_to_midi(p_est[n])
            fret = int(round(m_fq - librosa.note_to_midi(profile.tuning[s])))

            plt.scatter(t_st, s + 1, marker="${}$".format(fret), color='orange', label='Est.', s=200, alpha=0.75)

            if offset:
                plt.plot([t_st + 0.03, t_fn - 0.03], [s + 1] * 2, linestyle='-', color='orange', label='Est.', alpha=0.75)

    handles = [mlines.Line2D([], [], color='black', linestyle='-', label='Ref.', linewidth=3),
               mlines.Line2D([], [], color='orange', linestyle='-', label='Est.', linewidth=3)]

    plt.title(track_name)
    plt.xlabel('Time (s)')
    plt.ylabel('String')
    plt.yticks(range(1, 7), profile.tuning)
    plt.legend(handles=handles, loc='upper right', framealpha=0.5)
    plt.xlim(t_bounds[0] - 0.25, t_bounds[1] + 0.25)

    plt.gcf().set_size_inches(16, 4.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=500)
    else:
        plt.show()


def plot_pianoroll(pianoroll):
    #ax = plt.gca()

    pianoroll = pianoroll[..., 0 : 100]

    plt.imshow(pianoroll, cmap='gray_r', vmin=0, vmax=1)
    plt.gca().invert_yaxis()
    plt.ylabel('MIDI (R/E2)')
    plt.xlabel('Frame')
    #plt.grid()
    plt.show()

    return plt
