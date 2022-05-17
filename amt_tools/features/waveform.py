# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .common import FeatureModule

# Regular imports
import numpy as np


# TODO - could filterbank learning support in models/common.py be simplified?
# TODO - take another look at all of this to verify it all works as expected

class WaveformWrapper(FeatureModule):
    """
    Implements a audio waveform feature wrapper.
    """
    def __init__(self, sample_rate=44100, hop_length=512, decibels=False, win_length=None, center=True):
        """
        Initialize parameters for the waveform wrapper.

        Parameters
        ----------
        See FeatureModule class for others...
        win_length : int
          Number of samples to use for each frame;
          Must be less than or equal to n_fft;
          Defaults to n_fft if unspecified
        center : bool
          Whether to pad for centered frames
        """

        super().__init__(sample_rate=sample_rate,
                         hop_length=hop_length,
                         num_channels=1,
                         decibels=decibels)

        if win_length is None:
            win_length = self.hop_length
        self.win_length = win_length

        self.center = center

    def get_expected_frames(self, audio):
        """
        Determine the number of frames the module will return
        for a piece of audio.

        Parameters
        ----------
        audio : ndarray
          Mono-channel audio

        Returns
        ----------
        num_frames : int
          Number of frames expected
        """

        if self.center or audio.shape[-1] == 0:
            # Calculation is unchanged from default
            num_frames = super().get_expected_frames(audio)
        else:
            # The number of hops which have full frames, plus one for an incomplete frame
            num_frames = 1 + ((max(0, (audio.shape[-1] - self.win_length)) - 1) // self.hop_length + 1)

        return num_frames

    def get_sample_range(self, num_frames):
        """
        Determine the range of audio samples which will produce features
        with a given number of frames.

        Parameters
        ----------
        num_frames : int
          Number of frames for sample-range query

        Returns
        ----------
        sample_range : ndarray
          Valid audio signal lengths to obtain queried number of frames
        """

        if self.center or num_frames == 0:
            # Calculation is unchanged from default
            sample_range = super().get_sample_range(num_frames)
        else:
            if num_frames == 1:
                # Number of samples which will generate a full frame
                sample_range = np.arange(1, self.win_length + 1)
            else:
                # Number of hops which have full frames, plus one for an incomplete frame
                sample_range = np.arange(1, self.hop_length + 1) + \
                               self.get_num_samples_required() + (num_frames - 2) * self.hop_length

        return sample_range

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

    def _pad_audio(self, audio):
        """
        Pad audio such that is is divisible by the specified divisor.

        Parameters
        ----------
        audio : ndarray
          Mono-channel audio

        Returns
        ----------
        audio : ndarray
          Audio padded such that it will not throw away non-zero samples
        """

        if not self.center:
            # We need at least this many samples
            divisor = self.get_num_samples_required()
            if audio.shape[-1] > divisor:
                # After above is satisfied, just pad for one extra hop
                divisor = self.hop_length

            # Pad the audio
            audio = self.pad_audio(audio, divisor)

        return audio

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

        # TODO - is audio or None better?
        return None

    def get_times(self, audio):
        """
        Determine the time, in seconds, associated with frame.

        Parameters
        ----------
        audio: ndarray
          Mono-channel audio

        Returns
        ----------
        times : ndarray
          Time in seconds of each frame
        """

        times = super().get_times(audio)

        if not self.center:
            # Add a time offset equal to half the window length
            times += ((self.win_length // 2) / self.sample_rate)

        return times

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
