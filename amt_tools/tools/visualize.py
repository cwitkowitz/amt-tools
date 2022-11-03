# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from . import utils, constants

# Regular imports
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from abc import abstractmethod

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# TODO - move to pyqtgraph eventually?

__all__ = [
    'global_toolbar_disable',
    'initialize_figure',
    'get_dynamic_x_bounds',
    'get_dynamic_y_bounds',
    'Visualizer',
    'plot_waveform',
    'WaveformVisualizer',
    'plot_tfr',
    'TFRVisualizer',
    'plot_pitch_list',
    'plot_stacked_pitch_list',
    'StackedPitchListVisualizer',
    'plot_guitar_tablature',
    'GuitarTablatureVisualizer',
    'plot_pianoroll',
    'PianorollVisualizer',
    'plot_notes'
]


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

        # Request that the plot is redrawn
        self.fig.canvas.draw_idle()
        # Flush the GUI events for the figure
        self.fig.canvas.flush_events()

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


def plot_tfr(tfr, times=None, include_axes=True, fig=None):
    """
    Static function for plotting any time-frequency representation.

    Parameters
    ----------
    tfr : ndarray (F x T)
      Time-frequency representation
      F - number of frequency bins
      T - number of frames
    times : ndarray or None (Optional)
      Times corresponding to frames
    include_axes : bool
      Whether to include the axis in the visualizer
    fig : matplotlib Figure object
      Preexisting figure to use for plotting

    Returns
    ----------
    fig : matplotlib Figure object
      A handle for the figure used to plot the TFR
    """

    if fig is None:
        # Initialize a new figure if one was not given
        fig = initialize_figure(interactive=False)

    # Obtain a handle for the figure's current axis
    ax = fig.gca()

    if times is None:
        # Default times to ascending indices
        times = np.arange(tfr.shape[-1])

    # Determine the current time boundaries
    min_time = np.min(times)
    max_time = np.max(times)

    # Determine the number of bins in the TFR
    n_bins = tfr.shape[-2]

    # Check if a TFR has already been plotted
    if len(ax.images):
        # Update the TFR with the new data
        ax.images[0].set_data(tfr)
        # Determine the extent for the image
        extent = [min_time, max_time] + ax.images[0].get_extent()[-2:]
        # Update the images extent
        ax.images[0].set_extent(extent)
        # Flip the axis for ascending pitch
        ax.invert_yaxis()
    else:
        # Determine the extent for the image
        extent = [min_time, max_time, n_bins, 0]
        # Plot the tfr as an image
        ax.imshow(tfr, extent=extent)
        # Flip the axis for ascending pitch
        ax.invert_yaxis()
        # Make sure the image fills the figure
        ax.set_aspect('auto')

    if not include_axes:
        # Hide the axes
        ax.axis('off')
    else:
        # Add axis labels
        ax.set_ylabel('Frequency Bins')
        ax.set_xlabel('Time (s)')

    return fig


class TFRVisualizer(Visualizer):
    """
    Implements an iterative time-frequency representation (TFR) visualizer.
    """
    def __init__(self, figsize=None, include_axes=True, plot_frequency=None,
                 sample_rate=44100, hop_length=512, n_bins=192, time_window=1):
        """
        Initialize parameters for the TFR visualizer.

        Parameters
        ----------
        See Visualizer class for others...
        sample_rate : int or float
          Number of samples per second of audio
        hop_length : int or float
          Number of samples between feature frames
        n_bins : int
          Number of bins expected in TFR
        time_window : int or float
          Number of seconds to show in the plot at a time
        """

        super().__init__(figsize, include_axes, plot_frequency)

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_bins = n_bins

        # Determine the buffer size necessary for the chosen time window
        self.buffer_size = int(round(time_window * self.sample_rate / self.hop_length))

        # Buffer parameters
        self.time_buffer = None
        self.frame_buffer = None
        self.current_frame = None

        # Reset the visualizer
        self.reset()

    def update(self, frames):
        """
        Update the internal state of the TFR visualizer and display the updated plot.

        Parameters
        ----------
        frames : ndarray (F x T)
          New frames for the TFR
          F - number of frequency bins
          T - number of frames
        """

        # Determine how many new frames were given
        num_frames = frames.shape[-1]

        # Advance the frame buffer by the given amount of new frames
        self.frame_buffer = np.roll(self.frame_buffer, -num_frames)
        # Overwrite the oldest frames in the buffer with the new ones
        self.frame_buffer[..., -num_frames:] = frames

        # Obtain the corresponding times for the new samples
        times = np.arange(self.current_frame, self.current_frame + num_frames) / (self.sample_rate / self.hop_length)

        # Advance the time buffer by the given amount of new frames
        self.time_buffer = np.roll(self.time_buffer, -num_frames)
        # Overwrite the oldest times in the buffer with the new times
        self.time_buffer[-num_frames:] = times

        # Advance the time tracking frame index
        self.current_frame += num_frames

        if self.query_figure_update():
            self.pre_update()
            # Update the visualizer's figure
            self.fig = plot_tfr(tfr=self.frame_buffer,
                                times=self.time_buffer,
                                include_axes=self.include_axes,
                                fig=self.fig)
            self.post_update()

    def reset(self):
        """
        Reset the TFR visualizer.
        """

        # Clear any open figures, open a new one, and reset the frame counter
        super().reset()

        # Create a time buffer, where the very next sample added will be at time 0
        self.time_buffer = np.arange(-self.buffer_size, 0) / (self.sample_rate / self.hop_length)
        # Create a corresponding frame buffer
        self.frame_buffer = np.zeros((self.n_bins, self.buffer_size))
        # Reset the frame index used for tracking time
        self.current_frame = 0


def plot_pitch_list(times, pitch_list, hertz=False, point_size=5, marker='o', include_axes=True,
                    x_bounds=None, y_bounds=None, overlay=False, color='k', alpha=1.0, label=None,
                    idx=0, fig=None):
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
    marker : str
      Marker to use for the scatter points
    include_axes : bool
      Whether to include the axis in the visualizer
    x_bounds : list (length 2) of float or None (Optional)
      Lower and upper x-axis boundary to force
    y_bounds : list (length 2) of float or None (Optional)
      Lower and upper y-axis boundary to force
    overlay : bool
      Whether to overlay a new scatter plot rather than just modifying the data
    color : string
      Color for the pitch contour
    alpha : float in range [0, 1]
      Transparency of maximum activation
    label : string or None (Optional)
      Labels to use for legend
    idx : int
      Index for collection object (scatter plot) to re-use (if it exists)
    fig : matplotlib Figure object
      Preexisting figure to use for plotting

    Returns
    ----------
    fig : matplotlib Figure object
      A handle for the figure used to plot the pitch list
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
    if len(collections) and idx < len(collections) and not overlay:
        # Re-use the selected scatter collection and plot the new points
        collections[idx].set_offsets(np.c_[times, pitches])
    else:
        # Plot the points as a new collection
        ax.scatter(times, pitches, s=point_size, color=color, marker=marker, label=label, alpha=alpha)

    if y_bounds is None:
        # Get the dynamic y-axis boundaries if none were provided
        y_bounds = get_dynamic_y_bounds(ax, pitches)
    # Bound the y-axis
    ax.set_ylim(y_bounds)

    if x_bounds is None:
        # Get the dynamic x-axis boundaries if none were provided
        x_bounds = get_dynamic_x_bounds(ax, times)
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
      Whether to include the axis in the plot
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
      A handle for the figure used to plot the stacked pitch list
    """

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


def plot_guitar_tablature(stacked_frets, point_size=100, include_x_axis=True,
                          x_bounds=None, colors=None, labels=None, fig=None):
    """
    Static function for plotting guitar tablature organized as notes by string.
    TODO - this does not scale well for big note groups

    Parameters
    ----------
    stacked_frets : dict
      Dictionary containing (slice -> (pitches (fret), intervals)) pairs
    point_size : int or float
      Size of points within the scatter plot
    include_x_axis : bool
      Whether to include the time axis in the plot
    x_bounds : list (length 2) of float or None (Optional)
      Lower and upper x-axis boundary to force
    colors : list of string or None (Optional)
      Color (by string) for the notes
    labels : list of string or None (Optional)
      Labels for the strings
    fig : matplotlib Figure object
      Preexisting figure to use for plotting

    Returns
    ----------
    fig : matplotlib Figure object
      A handle for the figure used to plot the tablature
    """

    if fig is None:
        # Initialize a new figure if one was not given
        fig = initialize_figure(interactive=False, figsize=(10, 5))

    # Obtain a handle for the figure's current axis
    ax = fig.gca()

    if labels is None:
        # Default the string labels
        labels = constants.DEFAULT_GUITAR_LABELS

    # Set the y-axis ticks to the string labels
    ax.set_yticks(range(len(stacked_frets)))
    ax.set_yticklabels(labels)

    # Add some extra space to the y-axis
    y_padding = 0.5
    ax.set_ylim([-y_padding, len(stacked_frets) - y_padding])

    if x_bounds is not None:
        # Convert the notes to batched notes
        stacked_frets = utils.apply_func_stacked_representation(stacked_frets, utils.notes_to_batched_notes)
        # Remove notes occuring outside of the time-axis boundaries
        stacked_frets = utils.apply_func_stacked_representation(stacked_frets, utils.slice_batched_notes,
                                                                                  start_time=x_bounds[0],
                                                                                   stop_time=x_bounds[1])
        # Convert the batced notes back to note groups
        stacked_frets = utils.apply_func_stacked_representation(stacked_frets, utils.batched_notes_to_notes)

    # Obtain the labels for all lines
    line_labels = [line.get_label() for line in ax.get_lines()]

    # Construct a list to keep track of the notes plot
    labels_in_use = []

    # Loop through the stack of pitch lists, keeping track of the index
    for idx, slc in enumerate(stacked_frets.keys()):
        # Get the frets and intervals from the slice
        frets, intervals = stacked_frets[slc]

        # Make sure the frets are integers
        frets = frets.astype(constants.UINT)

        # Determine the color to use for the string
        color = 'k' if colors is None else colors[idx]

        for k, fret in enumerate(frets):
            # Obtain the onset for the note
            onset = intervals[k, 0]
            # Construct a unique label for the note
            label = f'{labels[idx]}_{fret}_{round(onset, 5)}'
            # Add the label to the tracked list
            labels_in_use += [label]

            if label not in line_labels:
                # Plot onset of the note as its fret number
                ax.scatter(onset, idx, marker="${}$".format(fret), color=color, label=label, s=point_size)
                # Plot the note durations as a line following the fret number
                ax.plot(intervals[k], [idx, idx], linestyle='-', color=color, label=label)
            else:
                # Find the existing line
                note_line = [line for line in ax.get_lines() if line.get_label() == label][0]
                # Extend the existing line
                note_line.set_xdata(intervals[k])

    # Obtain the onsets lines for the notes
    note_onsets = [point for point in ax.collections]
    note_lines = [line for line in ax.get_lines() if line.get_label() not in labels]

    # Loop through all of the lines on the plot
    for onset, line in zip(note_onsets, note_lines):
        # Check if the line is still in use
        if line.get_label() not in labels_in_use:
            # Remove the onset
            onset.remove()
            # Remove the line
            line.remove()

    # Obtain the lines for the string if they exist
    string_lines = [line for line in ax.get_lines() if line.get_label() in labels]

    if x_bounds is None:
        # Get the dynamic x-axis boundaries if none were provided
        #x_bounds = get_dynamic_x_bounds(ax, times)
        x_bounds = ax.get_xlim()
    # Bound the x-axis
    ax.set_xlim(x_bounds)

    # Loop through the strings
    for idx in range(len(stacked_frets)):
        if idx >= len(string_lines):
            # Plot the string for the first time
            ax.plot(x_bounds, [idx, idx], linewidth=1, color='k', label=f'{labels[idx]}', alpha=0.25)
        else:
            # Shift the line across time
            string_lines[idx].set_xdata(x_bounds)

    if not include_x_axis:
        # Hide the x-axis
        ax.axes.get_xaxis().set_visible(False)
    else:
        # Add x-axis labels
        ax.set_xlabel('Time (s)')

    return fig


class GuitarTablatureVisualizer(Visualizer):
    """
    Implements an iterative guitar tablature visualizer.
    """
    def __init__(self, figsize=None, include_axes=True, plot_frequency=None, time_window=1, colors=None, labels=None):
        """
        Initialize parameters for the guitar tablature visualizer.

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
        self.stacked_frets = None

        # Plotting parameters
        self.colors = colors
        self.labels = labels

        # Reset the visualizer
        self.reset()

    def update(self, current_time, stacked_frets):
        """
        Update the internal state of the guitar tablature visualizer and display the updated plot.

        Parameters
        ----------
        current_time : int
          Current time associated with the stacked frets
        stacked_frets : dict
          Dictionary containing (slice -> (frets, intervals)) pairs
        """

        # Determine the window of time which should be displayed
        time_window_start = current_time - self.time_window

        if self.stacked_frets is None:
            # Initialize the stacked pitch list with the given observations
            self.stacked_frets = stacked_frets
        else:
            # Concatenate the tracked stacked pitch list with the given observations
            self.stacked_frets = utils.cat_stacked_notes(self.stacked_frets, stacked_frets)
            self.stacked_frets = utils.filter_stacked_note_repeats(self.stacked_frets)

        # Slice the tracked stacked pitch list so observations are within the time window
        self.stacked_frets = utils.apply_func_stacked_representation(self.stacked_frets, utils.notes_to_batched_notes)
        self.stacked_frets = utils.apply_func_stacked_representation(self.stacked_frets, utils.slice_batched_notes,
                                                                     start_time=time_window_start,
                                                                     stop_time=current_time)
        self.stacked_frets = utils.apply_func_stacked_representation(self.stacked_frets, utils.batched_notes_to_notes)

        if self.query_figure_update():
            self.pre_update()
            # Set the x-axis boundaries to force
            x_bounds = [time_window_start, current_time]
            # Update the visualizer's figure
            self.fig = plot_guitar_tablature(stacked_frets=self.stacked_frets,
                                             include_x_axis=self.include_axes,
                                             x_bounds=x_bounds,
                                             colors=self.colors,
                                             labels=self.labels,
                                             fig=self.fig)
            self.post_update()

    def reset(self):
        """
        Reset the guitar tablature visualizer.
        """

        # Clear any open figures, open a new one, and reset the frame counter
        super().reset()

        # Clear the stacked frets buffer
        self.stacked_frets = None


def plot_pianoroll(multi_pitch, times=None, profile=None, include_axes=True,
                   overlay=False, color='k', alpha=1.0, fig=None):
    """
    Static function for plotting a 2D pitch activation map or pianoroll.

    Parameters
    ----------
    multi_pitch : ndarray (F x T)
      Discrete pitch activation map
      F - number of discrete pitches
      T - number of frames
    times : ndarray or None (Optional)
      Times corresponding to beginning of frames
    profile : InstrumentProfile (instrument.py)
      Instrument profile detailing experimental setup
    include_axes : bool
      Whether to include the axis in the visualizer
    overlay : bool
      Whether to overlay a new image on the plot rather than just modifying the data
    color : string
      Color for the pianoroll
    alpha : float in range [0, 1]
      Transparency of maximum activation
    fig : matplotlib Figure object
      Preexisting figure to use for plotting

    Returns
    ----------
    fig : matplotlib Figure object
      A handle for the figure used to plot the pianoroll
    """

    if fig is None:
        # Initialize a new figure if one was not given
        fig = initialize_figure(interactive=False)

    # Obtain a handle for the figure's current axis
    ax = fig.gca()

    if times is None:
        # Default times to ascending indices
        times = np.arange(multi_pitch.shape[-1])
        # Mark the x-axis as frame index
        x_label = 'Frame Index'
    else:
        # Mark the x-axis as time
        x_label = 'Time (s)'

    # Use the times (or frame indices) for the x-axis
    x_min, x_max = np.min(times), np.max(times) + utils.estimate_hop_length(times)

    # Use the relative pitch in the activation map for the y-axis
    y_min, y_max = -0.5, multi_pitch.shape[-2] - 0.5

    if profile is not None:
        # Add the offset of the lowest pitch
        y_min += profile.low
        y_max += profile.low

    # Set the extent for marking the axes of the image
    extent = [x_min, x_max, y_max, y_min]

    # Check if an activation map has already been plotted
    if len(ax.images) and not overlay:
        # Update the activation map with the new data
        ax.images[0].set_data(multi_pitch)
        # Update the images extent
        ax.images[0].set_extent(extent)
    else:
        # Construct the colormap
        cmap = LinearSegmentedColormap.from_list('', ['white', color], N=256)
        # Plot the activation map as an image
        ax.imshow(multi_pitch, cmap=cmap, vmin=0, vmax=1, extent=extent, aspect='auto', alpha=alpha)

    if ax.yaxis_inverted():
        # Flip the axis for ascending pitch
        ax.invert_yaxis()

    if not include_axes:
        # Hide the axes
        ax.axis('off')
    else:
        # Add axis labels
        y_label = 'Relative Pitch (MIDI)' if profile is None else 'Pitch (MIDI)'
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)

    return fig


class PianorollVisualizer(TFRVisualizer):
    """
    Implements an iterative pianoroll visualizer.
    """
    def __init__(self, figsize=None, include_axes=True, plot_frequency=None,
                 sample_rate=44100, hop_length=512, n_pitches=88, time_window=1):
        """
        Initialize parameters for the pianoroll visualizer.

        Parameters
        ----------
        See TFRVisualizer class for others...
        n_pitches : int
          Number of pitches expected in pianoroll
        """

        super().__init__(figsize, include_axes, plot_frequency, sample_rate, hop_length, n_pitches, time_window)

    def update(self, frames):
        """
        Update the internal state of the pianoroll visualizer and display the updated plot.

        Parameters
        ----------
        frames : ndarray (F x T)
          New frames for the pianoroll
          F - number of pitches
          T - number of frames
        """

        # Determine how many new frames were given
        num_frames = frames.shape[-1]

        # Advance the frame buffer by the given amount of new frames
        self.frame_buffer = np.roll(self.frame_buffer, -num_frames)
        # Overwrite the oldest frames in the buffer with the new ones
        self.frame_buffer[..., -num_frames:] = frames

        # Obtain the corresponding times for the new samples
        times = np.arange(self.current_frame, self.current_frame + num_frames) / (self.sample_rate / self.hop_length)

        # Advance the time buffer by the given amount of new frames
        self.time_buffer = np.roll(self.time_buffer, -num_frames)
        # Overwrite the oldest times in the buffer with the new times
        self.time_buffer[-num_frames:] = times

        # Advance the time tracking frame index
        self.current_frame += num_frames

        if self.query_figure_update():
            self.pre_update()
            # Update the visualizer's figure
            self.fig = plot_pianoroll(multi_pitch=self.frame_buffer,
                                      times=self.time_buffer,
                                      include_axes=self.include_axes,
                                      fig=self.fig)
            self.post_update()


def plot_notes(pitches, intervals, x_bounds=None, y_bounds=None, color='k', fig=None):
    """
    Static function for plotting note groups.

    TODO - make compatible w/ real-time note plotting like other visualization functions

    Parameters
    ----------
    pitches : ndarray (K)
      Array of pitches corresponding to notes in MIDI format
      (K - number of notes)
    intervals : ndarray (K x 2)
      Array of onset-offset time pairs corresponding to notes
    x_bounds : list (length 2) of float or None (Optional)
      Lower and upper x-axis boundary to force
    y_bounds : list (length 2) of float or None (Optional)
      Lower and upper y-axis boundary to force
    color : string
      Color for the note outlines
    fig : matplotlib Figure object
      Preexisting figure to use for plotting

    Returns
    ----------
    fig : matplotlib Figure object
      A handle for the figure used to plot the notes
    """

    if fig is None:
        # Initialize a new figure if one was not given
        fig = initialize_figure(interactive=False, figsize=(10, 5))

    # Obtain a handle for the figure's current axis
    ax = fig.gca()

    if y_bounds is None:
        # Get the dynamic y-axis boundaries if none were provided
        y_bounds = get_dynamic_y_bounds(ax, pitches)
    # Bound the y-axis
    ax.set_ylim(y_bounds)

    if x_bounds is None:
        # Get the dynamic x-axis boundaries if none were provided
        x_bounds = get_dynamic_x_bounds(ax, intervals.flatten())
    # Bound the x-axis
    ax.set_xlim(x_bounds)

    for p, i in zip(pitches, intervals):
        # Add a rectangle representing the note to the plot, applying an offset to center around the pitch
        ax.add_patch(Rectangle((i[0], p - 0.5), (i[1] - i[0]), 1, linewidth=1, edgecolor=color, facecolor='none'))

    # Add x-axis label
    ax.set_xlabel('Time (s)')
    # Ad y-axis label
    ax.set_ylabel('Pitch (MIDI)')

    return fig
