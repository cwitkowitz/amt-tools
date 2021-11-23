# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .common import FeatureModule

# Regular imports
import numpy as np


class WaveformWrapper(FeatureModule):
    """
    Implements a audio waveform feature wrapper.
    """
    def __init__(self, sample_rate=44100, hop_length=512, win_length=None):
        """
        Initialize parameters for the waveform wrapper.

        Parameters
        ----------
        See FeatureModule class for others...
        win_length : int
          Number of samples to use for each frame;
          Defaults to hop_length if unspecified
        """

        super().__init__(sample_rate, hop_length, 1, False)

        if win_length is None:
            win_length = self.hop_length
        self.win_length = win_length

    def get_num_samples_required(self):
        """
        Determine the number of samples required to extract one full frame of features.

        Returns
        ----------
        num_samples_required : int
          Number of samples
        """

        num_samples_required = self.win_length

        return num_samples_required

    def process_audio(self, audio):
        """
        Simply pass through the audio.

        Parameters
        ----------
        audio : ndarray
          Mono-channel audio

        Returns
        ----------
        audio : ndarray
          Mono-channel audio
        """

        return None

    def get_feature_size(self):
        """
        Helper function to access dimensionality of features.

        Returns
        ----------
        feature_size : int
          Dimensionality along feature axis
        """

        feature_size = self.win_length

        return feature_size

    def get_times(self, audio):
        """
        Determine the time, in seconds, associated with each sample.

        Parameters
        ----------
        audio: ndarray
          Mono-channel audio

        Returns
        ----------
        times : ndarray
          Time in seconds of each samples
        """

        # Obtain the (relative) time of each sample
        times = np.arange(len(audio)) / self.sample_rate

        return times
