# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .common import FeatureModule

# Regular imports
from librosa.util import frame

import numpy as np


# TODO - could filterbank learning support in models/common.py be simplified?

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
          Defaults to hop_length if unspecified
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

    def center_pad(self, audio):
        """
        Pad the audio such that the first sample
        is located halfway through the first frame.

        Parameters
        ----------
        audio : ndarray
          Mono-channel audio

        Returns
        ----------
        audio : ndarray
          Padded audio
        """

        # Compute the padding which would occur in (librosa) STFT
        padding = [tuple([int(self.win_length // 2)] * 2)]
        # Pad the signal on both sides
        audio = np.pad(audio, padding, mode='constant')

        return audio

    def process_audio(self, audio):
        """
        Chop the audio in frames according to window and hop length.

        Parameters
        ----------
        audio : ndarray
          Mono-channel audio

        Returns
        ----------
        audio_frames : ndarray
          Padded audio split into frames
        """

        if audio.shape[-1] == 0:
            # Handle case of empty audio array
            return np.zeros((self.win_length, 0))

        if self.center:
            # Pad the audio such that the first sample
            # is located halfway through the first frame
            audio = self.center_pad(audio)
        else:
            # Pad the audio to fill in a final frame
            audio = self.frame_pad(audio)

        # Obtain the audio samples associated with each frame
        audio_frames = frame(audio,
                             frame_length=self.win_length,
                             hop_length=self.hop_length)

        return audio_frames

    def get_times(self, audio, at_start=False):
        """
        Determine the time, in seconds, associated with each frame.

        Parameters
        ----------
        audio: ndarray
          Mono-channel audio
        at_start : bool
          Whether time is associated with beginning of frame instead of center

        Returns
        ----------
        times : ndarray
          Time in seconds of each frame
        """

        # Get the times associated with the hops
        times = super().get_times(audio)

        if self.center and at_start:
            # Subtract half of the window length
            times -= ((self.win_length // 2) / self.sample_rate)
        elif not self.center and not at_start:
            # Add half of the window length
            times += ((self.win_length // 2) / self.sample_rate)
        else:
            # Simply the hop times we already have
            pass

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
