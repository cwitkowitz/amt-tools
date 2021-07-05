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

# TODO - move to pyqtgraph eventually?


def global_toolbar_disable():
    """
    Change default matplotlib parameter so no toolbar is added to plots.
    """

    # TODO - ideally there is a better way to do this
    mpl.rcParams['toolbar'] = 'None'


def initialize_figure(figsize=None, interactive=False):
    """
    Create a new figure and display it.

    Parameters
    ----------
    figsize : tuple (x, y) or None (Optional)
      Size of plot window in inches - if unspecified set to default
    interactive : bool
      Whether to set turn on matplotlib interactive mode

    Returns
    ----------
    fig : matplotlib Figure object
      A handle for the created figure
    """

    if interactive and not plt.isinteractive():
        # Make sure pyplot is in interactive mode
        plt.ion()

    # Create a new figure with the specified size
    fig = plt.figure(figsize=figsize, tight_layout=True)

    if not interactive:
        # Open the figure manually if interactive mode is off
        plt.show(block=False)

    return fig


def get_dynamic_x_bounds(ax, x_values, scale_factor=1.0):
    """
    Compute new x-axis boundaries for a dynamic plot.

    Parameters
    ----------
    ax : matplotlib AxesSubplot
      Axis for which to get the bounds
    x_values : ndarray
      Current x-axis values for the plot
    scale_factor : float
      Value by which to scale the boundaries

    Returns
    ----------
    x_bounds : list (length 2) of float
      Lower and upper x-axis boundary to force
    """

    # Get the current boundaries
    x_bounds = ax.get_xlim()

    # Check if there are values to compare
    if len(x_values) > 0:
        # Set the upper bound to the maximum between the current bound and the scaled highest x value
        upper_x_bound = max(x_bounds[1], scale_factor * np.max(x_values))
        # Set the lower bound to scaled lowest x value
        lower_x_bound = scale_factor * np.min(x_values)
        # Provide the new boundaries
        x_bounds = [lower_x_bound, upper_x_bound]

    return x_bounds


def get_dynamic_y_bounds(ax, y_values, scale_factor=1.05):
    """
    Compute new y-axis boundaries for a dynamic plot.

    Parameters
    ----------
    ax : matplotlib AxesSubplot
      Axis for which to get the bounds
    y_values : ndarray
      Current y-axis values for the plot
    scale_factor : float
      Value by which to scale the boundaries

    Returns
    ----------
    y_bounds : list (length 2) of float
      Lower and upper y-axis boundary to force
    """

    # Get the current boundaries
    y_bounds = ax.get_ylim()

    # Check if there are values to compare
    if len(y_values) > 0:
        # Set the lower bound to the minimum between the current bound and the scaled lowest y value
        lower_y_bound = min(y_bounds[0], scale_factor * np.min(y_values))
        # Set the upper bound to the maximum between the current bound and the scaled highest y value
        upper_y_bound = max(y_bounds[1], scale_factor * np.max(y_values))
        # Provide the new boundaries
        y_bounds = [lower_y_bound, upper_y_bound]

    return y_bounds


class Visualizer(object):
    """
    Implements a generic iterative visualizer.
    """

    def __init__(self, figsize=None, include_axes=True, plot_frequency=None):
        """
        Initialize parameters common to all visualizers.

        Parameters
        ----------
        figsize : tuple (x, y) or None (Optional)
          Size of plot window in inches - if unspecified set to default
        include_axes : bool
          Whether to display the axes in the visualizer
        plot_frequency : int or None
          N, where every Nth update() call results in a plot update
        """

        # Plotting parameters
        self.figsize = figsize
        self.include_axes = include_axes

        # Set the plotting frequency
        if plot_frequency is None:
            # Default the plotting frequency to 1 (update plot every time)
            plot_frequency = 1
        self.plot_frequency = plot_frequency

        # Visualizer parameters
        self.fig = None
        self.frame_counter = None

    def increment_frame_count(self):
        """
        Increase the frame count by one.
        """

        if self.frame_counter is None:
            # Initialize the frame counter
            self.frame_counter = 1
        else:
            # Increase the frame counter by 1
            self.frame_counter += 1

    def query_figure_update(self):
        """
        Determine if the figure should be updated during this iteration of update().

        Returns
        ----------
        plot_update : bool
          Flag indicating figure should be updated
        """

        # Default the flag
        figure_update = True

        # Check if the plot frequency less than every time
        if self.plot_frequency > 1:
            # Update the frame count
            self.increment_frame_count()

            # Check if the frame count doesn't go evenly into the plot frequency
            if self.frame_counter % self.plot_frequency:
                # Set the flag
                figure_update = False

        return figure_update

    @abstractmethod
    def update(self):
        """
        Update the internal state of the visualizer and the plot.
        """

        return NotImplementedError

    def pre_update(self):
        """
        Perform any steps prior to updating the plot.
        """

        pass

    def post_update(self):
        """
        Perform any steps after updating the plot.
        """

        t = utils.get_current_time()
        # Request that the plot is redrawn
        self.fig.canvas.draw_idle()
        # Flush the GUI events for the figure
        self.fig.canvas.flush_events()
        #utils.compute_time_difference(t, True, 'Post Update')

    def close(self):
        """
        Close the visualizer's figure, if it exists.
        """

        # Check if the figure exists
        if self.fig is not None:
            # Close the figure
            plt.close(self.fig)

        self.fig = None

    @abstractmethod
    def reset(self):
        """
        Reset the visualizer.
        """

        # Close the current figure, if one exists
        self.close()

        # Reset the frame counter
        self.frame_counter = None

        # Re-initialize a figure
        self.fig = initialize_figure(figsize=self.figsize, interactive=True)


def plot_waveform(samples, times=None, include_axes=True, y_bounds=None, color='k', fig=None):
    """
    Static function for plotting a 1D waveform.

    Parameters
    ----------
    samples : ndarray
      Waveform samples to plot
    times : ndarray or None (Optional)
      Times corresponding to waveform samples
    include_axes : bool
      Whether to include the axis in the visualizer
    y_bounds : list (length 2) of float or None (Optional)
      Lower and upper y-axis boundary to force
    color : string
      Color for the waveform
    fig : matplotlib Figure object
      Preexisting figure to use for plotting

    Returns
    ----------
    fig : matplotlib Figure object
      A handle for the figure used to plot the waveform
    """

    if fig is None:
        # Initialize a new figure if one was not given
        fig = initialize_figure(interactive=False)

    # Obtain a handle for the figure's current axis
    ax = fig.gca()

    if times is None:
        # Default times to ascending indices
        times = np.arange(len(samples))

    # Check if a line has already been plotted
    lines = ax.get_lines()
    if len(lines):
        # Just change the data of the existing line
        lines[0].set_xdata(times)
        lines[0].set_ydata(samples)
    else:
        # Plot the waveform as a new line
        ax.plot(times, samples, color=color)

    # Bound the x-axis to the range given by the times
    ax.set_xlim([np.min(times), np.max(times)])

    if y_bounds is None:
        # Get the dynamic y-axis boundaries if none were provided
        y_bounds = get_dynamic_y_bounds(ax, samples)
    # Bound the y-axis
    ax.set_ylim(y_bounds)

    if not include_axes:
        # Hide the axes
        ax.axis('off')
    else:
        # Add axis labels
        ax.set_ylabel('A')
        ax.set_xlabel('Time (s)')

    return fig


class WaveformVisualizer(Visualizer):
    """
    Implements an iterative waveform visualizer.
    """
    def __init__(self, figsize=None, include_axes=True, plot_frequency=None, sample_rate=44100, time_window=1):
        """
        Initialize parameters for the waveform visualizer.

        Parameters
        ----------
        See Visualizer class for others...
        sample_rate : int or float
          Number of samples per second of audio
        time_window : int or float
          Number of seconds to show in the plot at a time
        """

        super().__init__(figsize, include_axes, plot_frequency)

        self.sample_rate = sample_rate
        # Determine the buffer size necessary for the chosen time window
        self.buffer_size = int(round(time_window * self.sample_rate))

        # Buffer parameters
        self.time_buffer = None
        self.sample_buffer = None
        self.current_sample = None

        # Reset the visualizer
        self.reset()

    def update(self, samples):
        """
        Update the internal state of the waveform visualizer and display the updated plot.

        Parameters
        ----------
        samples : ndarray
          New waveform samples to track
        """

        # Determine how many new samples were given
        num_samples = len(samples)

        # Advance the audio buffer by the given amount of new samples
        self.sample_buffer = np.roll(self.sample_buffer, -num_samples)
        # Overwrite the oldest samples in the buffer with the new audio
        self.sample_buffer[-num_samples:] = samples

        # Obtain the corresponding times for the new samples
        times = np.arange(self.current_sample, self.current_sample + num_samples) / self.sample_rate

        # Advance the time buffer by the given amount of new samples
        self.time_buffer = np.roll(self.time_buffer, -num_samples)
        # Overwrite the oldest times in the buffer with the new times
        self.time_buffer[-num_samples:] = times

        # Advance the time tracking sample index
        self.current_sample += num_samples

        if self.query_figure_update():
            self.pre_update()
            # Update the visualizer's figure
            self.fig = plot_waveform(samples=self.sample_buffer,
                                     times=self.time_buffer,
                                     include_axes=self.include_axes,
                                     fig=self.fig)
            self.post_update()

    def reset(self):
        """
        Reset the waveform visualizer.
        """

        # Clear any open figures, open a new one, and reset the frame counter
        super().reset()

        # Create a time buffer, where the very next sample added will be at time 0
        self.time_buffer = np.arange(-self.buffer_size, 0) / self.sample_rate
        # Create a corresponding sample buffer
        self.sample_buffer = np.zeros(self.buffer_size)
        # Reset the sample index used for tracking time
        self.current_sample = 0


def plot_pitch_list(times, pitch_list, hertz=False, point_size=5, include_axes=True,
                    x_bounds=None, y_bounds=None, color='k', label=None, idx=0, fig=None):
    """
    Static function for plotting pitch contours (pitch_list).

    Parameters
    ----------
    times : ndarray (N)
      Time in seconds of beginning of each frame
      N - number of time samples (frames)
    pitch_list : list of ndarray (N x [...])
      Array of pitches active during each frame
      N - number of pitch observations (frames)
    hertz : bool
      Whether to expect pitches in Hertz as opposed to MIDI
    point_size : int or float
      Size of points within the scatter plot
    include_axes : bool
      Whether to include the axis in the visualizer
    x_bounds : list (length 2) of float or None (Optional)
      Lower and upper x-axis boundary to force
    y_bounds : list (length 2) of float or None (Optional)
      Lower and upper y-axis boundary to force
    color : string
      Color for the pitch contour
    label : string or None (Optional)
      Labels to use for legend
    idx : int
      Index for collection object (scatter plot) to re-use (if it exists)
    fig : matplotlib Figure object
      Preexisting figure to use for plotting

    Returns
    ----------
    fig : matplotlib Figure object
      A handle for the figure used to plot the waveform
    """

    # Unroll the pitch list so it can be plotted
    times, pitches = utils.unroll_pitch_list(times, pitch_list)

    if fig is None:
        # Initialize a new figure if one was not given
        fig = initialize_figure(interactive=False)

    # Obtain a handle for the figure's current axis
    ax = fig.gca()

    # Check if scatter collections have already been plotted
    collections = ax.collections
    if len(collections) and idx < len(collections):
        # Re-use the selected scatter collection and plot the new points
        collections[idx].set_offsets(np.c_[times, pitches])
    else:
        # Plot the points as a new collection
        ax.scatter(times, pitches, s=point_size, color=color, label=label)

    if y_bounds is None:
        # Get the dynamic y-axis boundaries if none were provided
        y_bounds = get_dynamic_y_bounds(ax, pitches)
    # Bound the y-axis
    ax.set_ylim(y_bounds)

    if x_bounds is None:
        # Get the dynamic x-axis boundaries if none were provided
        x_bounds = get_dynamic_y_bounds(ax, times)
    # Bound the x-axis
    ax.set_xlim(x_bounds)

    if not include_axes:
        # Hide the axes
        ax.axis('off')
    else:
        # Create the label for frequency units
        units = 'Hz' if hertz else 'MIDI'
        # Add axis labels
        ax.set_ylabel(f'Pitch ({units})')
        ax.set_xlabel('Time (s)')

    if label is not None:
        # Show the legend
        ax.legend()

    return fig


def plot_stacked_pitch_list(stacked_pitch_list, hertz=False, point_size=5, include_axes=True,
                            x_bounds=None, y_bounds=None, colors=None, labels=None, fig=None):
    """
    Static function for plotting pitch contours (pitch_list).

    Parameters
    ----------
    stacked_pitch_list : dict
      Dictionary containing (slice -> (times, pitch_list)) pairs
    hertz : bool
      Whether to expect pitches in Hertz as opposed to MIDI
    point_size : int or float
      Size of points within the scatter plot
    include_axes : bool
      Whether to include the axis in the visualizer
    x_bounds : list (length 2) of float or None (Optional)
      Lower and upper x-axis boundary to force
    y_bounds : list (length 2) of float or None (Optional)
      Lower and upper y-axis boundary to force
    colors : list of string or None (Optional)
      Color for the pitch contour
    labels : list of string or None (Optional)
      Labels to use for legend
    fig : matplotlib Figure object
      Preexisting figure to use for plotting

    Returns
    ----------
    fig : matplotlib Figure object
      A handle for the figure used to plot the waveform
    """

    t = utils.get_current_time()
    # Loop through the stack of pitch lists, keeping track of the index
    for idx, slc in enumerate(stacked_pitch_list.keys()):
        # Get the times and pitches from the slice
        times, pitch_list = stacked_pitch_list[slc]
        # Determine the color to use when plotting the slice
        color = 'k' if colors is None else colors[idx]
        # Determine the label to use when plotting the slice
        label = None if labels is None else labels[idx]

        # Use the pitch_list plotting function
        fig = plot_pitch_list(times=times,
                              pitch_list=pitch_list,
                              hertz=hertz,
                              point_size=point_size,
                              include_axes=include_axes,
                              x_bounds=x_bounds,
                              y_bounds=y_bounds,
                              color=color,
                              label=label,
                              idx=idx,
                              fig=fig)
    #utils.compute_time_difference(t, True, 'Figure Update')

    return fig


class StackedPitchListVisualizer(Visualizer):
    """
    Implements an iterative stacked pitch list visualizer.
    """
    def __init__(self, figsize=None, include_axes=True, plot_frequency=None, time_window=1, colors=None, labels=None):
        """
        Initialize parameters for the stacked pitch list visualizer.

        Parameters
        ----------
        See Visualizer class for others...
        time_window : int or float
          Number of seconds to show in the plot at a time
        colors : list of string or None (Optional)
          Color for the pitch contour
        labels : list of string or None (Optional)
          Labels to use for legend
        """

        super().__init__(figsize, include_axes, plot_frequency)

        # Buffer parameters
        self.time_window = time_window
        self.stacked_pitch_list = None

        # Plotting parameters
        self.colors = colors
        self.labels = labels

        # Reset the visualizer
        self.reset()

    def update(self, current_time, stacked_pitch_list):
        """
        Update the internal state of the stacked pitch list visualizer and display the updated plot.

        Parameters
        ----------
        current_time : int
          Current time associated with the stacked pitch list
        stacked_pitch_list : dict
          Dictionary containing (slice -> (times, pitch_list)) pairs
        """

        # Determine the window of time which should be displayed
        time_window_start = current_time - self.time_window

        if self.stacked_pitch_list is None:
            # Initialize the stacked pitch list with the given observations
            self.stacked_pitch_list = stacked_pitch_list
        else:
            # Concatenate the tracked stacked pitch list with the given observations
            self.stacked_pitch_list = utils.cat_stacked_pitch_list(self.stacked_pitch_list, stacked_pitch_list)

        # Slice the tracked stacked pitch list so observations are within the time window
        self.stacked_pitch_list = utils.slice_stacked_pitch_list(self.stacked_pitch_list,
                                                                 time_window_start,
                                                                 current_time)

        if self.query_figure_update():
            self.pre_update()
            # Set the x-axis boundaries to force
            x_bounds = [time_window_start, current_time]
            # Update the visualizer's figure
            self.fig = plot_stacked_pitch_list(stacked_pitch_list=self.stacked_pitch_list,
                                               include_axes=self.include_axes,
                                               x_bounds=x_bounds,
                                               colors=self.colors,
                                               labels=self.labels,
                                               fig=self.fig)
            self.post_update()

    def reset(self):
        """
        Reset the stacked pitch list visualizer.
        """

        # Clear any open figures, open a new one, and reset the frame counter
        super().reset()

        # Clear the stacked pitch list buffer
        self.stacked_pitch_list = None


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
