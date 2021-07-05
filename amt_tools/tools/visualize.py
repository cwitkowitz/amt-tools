# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from . import utils

# Regular imports
from abc import abstractmethod

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import librosa

# TODO - move to pyqtgraph


def global_toolbar_disable():
    # TODO - ideally there is a better way to do this
    mpl.rcParams['toolbar'] = 'None'


class Visualizer(object):
    """
    Implements a generic visualizer.
    """

    def __init__(self, figsize=None, include_axis=True, plot_frequency=None):
        """
        Initialize parameters common to all visualizers.

        Parameters
        ----------
        TODO
        """

        # Plotting parameters
        # TODO - these parameters here or in plot call?
        self.include_axis = include_axis

        if plot_frequency is None:
            plot_frequency = 1
        self.plot_frequency = plot_frequency

        if self.plot_frequency > 1:
            self.frame_counter = 0

        self.fig = self.initialize_figure(figsize, True)

    @staticmethod
    def initialize_figure(figsize=None, interactive=False):
        if interactive and not plt.isinteractive():
            # Make sure pyplot is in interactive mode
            plt.ion()

        # Create a new figure
        fig = plt.figure(figsize=figsize, tight_layout=True)

        if not interactive:
            plt.show(block=False)

        return fig

    def query_plot(self):
        flag = False
        if self.plot_frequency > 1:
            self.frame_counter += 1

            if self.frame_counter % self.plot_frequency:
                flag = False
            else:
                flag = True

        return flag

    @abstractmethod
    def update(self):
        return NotImplementedError

    @abstractmethod
    def visualize(self):
        return NotImplementedError

    def pre_update(self):
        # TODO - don't do this - slows down significantly
        #self.fig.clear()
        pass

    def post_update(self):
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # Remove extraneous whitespace
        #self.fig.tight_layout()

    def kill(self):
        if self.fig is not None:
            plt.close(self.fig)

        self.fig = None


def plot_waveform(samples, times=None, include_axes=True, ylim=None, color='k', fig=None):
    """
    Static function for plotting a 1D waveform.

    TODO
    """

    if fig is None:
        fig = Visualizer.initialize_figure(interactive=False)

    ax = fig.gca()

    if times is None:
        times = np.arange(len(samples))

    lines = ax.get_lines()

    if len(lines):
        lines[0].set_xdata(times)
        lines[0].set_ydata(samples)
    else:
        ax.plot(times, samples, color=color)

    ax.set_xlim([np.min(times), np.max(times)])

    if ylim is None:
        ylim = ax.get_ylim()
        min_y = min(ylim[0], 1.05 * np.min(samples))
        max_y = max(ylim[1], 1.05 * np.max(samples))
        ylim = [min_y, max_y]
    ax.set_ylim(ylim)

    if not include_axes:
        ax.axis('off')
    else:
        ax.set_ylabel('A')
        ax.set_xlabel('Time (s)')

    return fig


class WaveformVisualizer(Visualizer):
    """
    Implements an iterative waveform visualizer.
    """
    def __init__(self, figsize=None, include_axis=True, plot_frequency=None, sample_rate=44100, time_window=1):
        super().__init__(figsize, include_axis, plot_frequency)

        self.sample_rate = sample_rate

        buffer_size = int(round(time_window * sample_rate))

        self.sample_buffer = np.zeros(buffer_size)
        self.time_buffer = np.arange(-buffer_size, 0) / self.sample_rate

        self.current_sample = 0

    def update(self, samples):

        num_samples = len(samples)

        self.sample_buffer = np.roll(self.sample_buffer, -num_samples)
        self.sample_buffer[-num_samples:] = samples

        times = np.arange(self.current_sample, self.current_sample + num_samples) / self.sample_rate
        self.current_sample += num_samples

        self.time_buffer = np.roll(self.time_buffer, -num_samples)
        self.time_buffer[-num_samples:] = times

        if not self.query_plot():
            return

        self.pre_update()
        self.fig = plot_waveform(self.sample_buffer, self.time_buffer, fig=self.fig)
        t = utils.get_current_time()
        self.post_update()
        utils.compute_time_difference(t, True, 'Post Update')


def plot_pitch_list(times, pitch_list, include_axes=True, xlim=None, color='k', k=0, fig=None):
    if fig is None:
        fig = Visualizer.initialize_figure(interactive=False)

    times = np.concatenate([[times[i]] * len(pitch_list[i]) for i in range(len(pitch_list))])
    pitches = np.concatenate(pitch_list)

    ax = fig.gca()

    collections = ax.collections

    if len(collections) and k < len(collections):
        collections[k].set_offsets(np.c_[times, pitches])
    else:
        ax.scatter(times, pitches, s=5, color=color)

    if xlim is not None:
        ax.set_xlim(xlim)

    # TODO - turn into function clip_y()
    ylim = ax.get_ylim()
    if len(pitches) > 0:
        min_y = min(ylim[0], 1.05 * np.min(pitches))
        max_y = max(ylim[1], 1.05 * np.max(pitches))
        ylim = [min_y, max_y]
    ax.set_ylim(ylim)

    if not include_axes:
        ax.axis('off')
    else:
        ax.set_ylabel('Pitch (MIDI)')
        ax.set_xlabel('Time (s)')

    return fig


def plot_stacked_pitch_list(stacked_pitch_list, include_axes=True, xlim=None, colors=None, fig=None):
    """
    TODO
    """

    if fig is None:
        fig = Visualizer.initialize_figure(interactive=False)

    # Loop through the stack of pitch lists
    for i, slc in enumerate(stacked_pitch_list.keys()):
        # Get the times and pitches from the slice
        times, pitch_list = stacked_pitch_list[slc]
        color = 'k' if colors is None else colors[i]
        fig = plot_pitch_list(times, pitch_list, include_axes, xlim, color=color, k=i, fig=fig)

    #ax = fig.gca()

    #if times is None:
    #    times = np.arange(len(samples))

    #lines = ax.get_lines()

    #if len(lines):
    #    lines[0].set_xdata(times)
    #    lines[0].set_ydata(samples)
    #else:
    #    ax.plot(times, samples, color=color)

    #ax.set_xlim([np.min(times), np.max(times)])

    #if not ylim:
    #    ylim = ax.get_ylim()
    #    min_y = min(ylim[0], 1.05 * np.min(samples))
    #    max_y = max(ylim[1], 1.05 * np.max(samples))
    #    ylim = [min_y, max_y]
    #ax.set_ylim(ylim)

    #if not include_axes:
    #    ax.axis('off')
    #else:
    #    ax.set_ylabel('Pitch (MIDI)')
    #    ax.set_xlabel('Time (s)')

    return fig


class StackedPitchListVisualizer(Visualizer):
    """
    Implements an iterative stacked pitch list visualizer.
    """
    def __init__(self, figsize=None, include_axis=True, plot_frequency=None, time_window=1):
        super().__init__(figsize, include_axis, plot_frequency)

        self.time_window = time_window

        self.stacked_pitch_list = None

    def update(self, current_time, stacked_pitch_list):
        time_window_start = current_time - self.time_window

        if self.stacked_pitch_list is None:
            self.stacked_pitch_list = stacked_pitch_list
        else:
            self.stacked_pitch_list = utils.cat_stacked_pitch_list(self.stacked_pitch_list, stacked_pitch_list)

        self.stacked_pitch_list = utils.slice_stacked_pitch_list(self.stacked_pitch_list,
                                                                 time_window_start,
                                                                 current_time)

        if not self.query_plot():
            return

        self.pre_update()
        xlim = [time_window_start, current_time]
        self.fig = plot_stacked_pitch_list(self.stacked_pitch_list, xlim=xlim, fig=self.fig)
        self.post_update()


def visualize_multi_pitch(multi_pitch, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.imshow(multi_pitch, cmap='gray_r', vmin=0, vmax=1)
    ax.invert_yaxis()

    return ax

# TODO - this is mostly trash for now - I've yet to make an effort to reestablish this
# TODO - see earlier commits to get started


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
